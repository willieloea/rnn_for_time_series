import io
import math
import unittest

import torch

from assignment3.losses import masked_mse
from assignment3.models import ElmanRNN, JordanRNN, MultiRNN


MODELS = (ElmanRNN, JordanRNN, MultiRNN)


class RecurrentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_explicit_scalar_recurrences(self):
        # Distinct effective feedback gains: 0.5 (Elman), 1 (Jordan), 1.5 (both).
        for model_type, gain in zip(MODELS, [0.5, 1.0, 1.5]):
            with self.subTest(model=model_type.__name__):
                model = model_type(1, 1, 1).double()
                with torch.no_grad():
                    model.input_to_hidden.weight.fill_(1)
                    model.input_to_hidden.bias.zero_()
                    model.hidden_to_output.weight.fill_(2)
                    model.hidden_to_output.bias.zero_()
                    for feedback in [model.hidden_to_hidden, model.output_to_hidden]:
                        if feedback is not None:
                            feedback.weight.fill_(0.5)
                hidden, expected = 0., []
                for value in [0.2, -0.1, 0.3]:
                    hidden = math.tanh(value + gain * hidden)
                    expected.append(2 * hidden)
                x = torch.tensor([[[0.2], [-0.1], [0.3]]], dtype=torch.float64)
                actual = model(x, return_sequence=True)
                torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64).reshape(1, 3, 1))
                torch.testing.assert_close(model(x), actual[:, -1])

    def test_batches_are_independent_and_calls_reset_context(self):
        for model_type in MODELS:
            model = model_type(4, 8, 2)
            x = torch.randn(3, 7, 4)
            expected = model(x)
            self.assertEqual(expected.shape, (3, 2))
            model(torch.randn(5, 20, 4))
            torch.testing.assert_close(model(x), expected)
            individual = torch.cat([model(example.unsqueeze(0)) for example in x])
            torch.testing.assert_close(individual, expected)

    def test_gradients_reach_early_inputs_and_feedback_weights(self):
        for model_type in MODELS:
            torch.manual_seed(17)
            model = model_type(2, 4, 2).double()
            x = (torch.randn(2, 4, 2, dtype=torch.float64) * 0.1).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(model, (x,)))
            model(x).square().mean().backward()
            self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_parameter_counts(self):
        # One hidden bias and one output bias. Input=4, hidden=8, output=2.
        base = 4 * 8 + 8 + 8 * 2 + 2
        for model_type, additional in zip(MODELS, [8 * 8, 2 * 8, 8 * 8 + 2 * 8]):
            self.assertEqual(model_type(4, 8, 2).parameter_count, base + additional)

    def test_seed_reproducibility_and_save_reload(self):
        for model_type in MODELS:
            torch.manual_seed(12)
            first = model_type(2, 4, 1)
            torch.manual_seed(12)
            second = model_type(2, 4, 1)
            x = torch.randn(3, 5, 2)
            torch.testing.assert_close(first(x), second(x))
            buffer = io.BytesIO()
            torch.save(first.state_dict(), buffer)
            buffer.seek(0)
            restored = model_type(2, 4, 1)
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(first(x), restored(x))

    def test_masked_loss_has_zero_gradient_for_unobserved_targets(self):
        prediction = torch.tensor([[2., 100.], [4., 5.]], requires_grad=True)
        target = torch.tensor([[1., float('nan')], [2., 3.]])
        mask = torch.tensor([[True, False], [True, True]])
        loss = masked_mse(prediction, target, mask)
        self.assertEqual(loss.item(), 3.)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[2/3, 0.], [4/3, 4/3]]))
        with self.assertRaises(ValueError):
            masked_mse(prediction, target, torch.zeros_like(mask))

    def test_invalid_dimensions(self):
        for model_type in MODELS:
            with self.assertRaises(ValueError):
                model_type(1, 0, 1)
            model = model_type(2)
            for x in [torch.zeros(4, 2), torch.zeros(4, 0, 2), torch.zeros(4, 5, 3),
                      torch.ones(4, 5, 2, dtype=torch.int64)]:
                with self.assertRaises(ValueError):
                    model(x)

    def test_learns_small_sequence_regression(self):
        # A controlled learning check, not a claim about held-out forecasting.
        torch.manual_seed(3)
        x = torch.randn(64, 4, 1) * 0.4
        target = 0.4 * x[:, 0] + 0.6 * x[:, -1]
        mask = torch.ones_like(target, dtype=torch.bool)
        for model_type in MODELS:
            torch.manual_seed(8)
            model = model_type(1, 8, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            initial = masked_mse(model(x), target, mask).item()
            for _ in range(150):
                optimizer.zero_grad()
                loss = masked_mse(model(x), target, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            final = masked_mse(model(x), target, mask).item()
            self.assertLess(final, initial * 0.2, model_type.__name__)


if __name__ == '__main__':
    unittest.main()
