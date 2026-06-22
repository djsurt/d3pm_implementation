"""Stage 5 unit tests — the training loop and checkpoint roundtrip.

These run a *tiny* real training on the `cleaman` smoke-test family (small Config,
few files, a couple of epochs, on CPU) and assert the contract Stage 6 relies on:

  * `train` returns a per-epoch loss list and writes the three checkpoint files;
  * `load_checkpoint` rebuilds the exact same architecture and loads every weight
    (state_dict keys match, weights identical, logits identical in eval mode);
  * the reloaded `Vocabulary` and `AbsorbingDiffusion` carry the right config.

We keep this fast (it is a correctness check, not a quality check — loss
*magnitude* is exercised by verify/verify_train.py), so we do NOT assert the loss
drops here; that needs more epochs and would make the test slow/flaky.
"""

from pathlib import Path

import pytest
import torch

from d3pm.config import Config
from d3pm.train import load_checkpoint, train


MALICIA = Path(__file__).resolve().parent.parent / "malicia"
FAMILY = "cleaman"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once (tiny) and reuse the checkpoint across tests in this module."""
    if not (MALICIA / FAMILY).is_dir():
        pytest.skip(f"malicia/{FAMILY} not present")
    out_dir = tmp_path_factory.mktemp("checkpoints")
    cfg = Config.small()
    cfg.T = 50
    cfg.max_len = 64
    summary = train(
        FAMILY,
        malicia_dir=str(MALICIA),
        out_dir=str(out_dir),
        max_files=8,
        epochs=2,
        batch_size=4,
        cfg=cfg,
        lr=1e-3,
        seed=0,
        dev=torch.device("cpu"),
        verbose=False,
    )
    return out_dir, cfg, summary


def test_train_returns_loss_history(trained):
    _, _, summary = trained
    losses = summary["epoch_losses"]
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert summary["num_chunks"] > 0


def test_checkpoint_files_written(trained):
    out_dir, _, _ = trained
    fam = out_dir / FAMILY
    assert (fam / "d3pm.pt").is_file()
    assert (fam / "vocab.pkl").is_file()
    assert (fam / "losses.json").is_file()


def test_load_checkpoint_weights_match(trained):
    out_dir, _, _ = trained
    model, diffusion, vocab, meta = load_checkpoint(
        FAMILY, out_dir=str(out_dir), dev=torch.device("cpu")
    )
    saved = torch.load(out_dir / FAMILY / "d3pm.pt", map_location="cpu")["model_state"]
    reloaded = model.state_dict()
    assert set(saved) == set(reloaded)
    for k in saved:
        assert torch.equal(saved[k], reloaded[k]), f"weight mismatch at {k}"


def test_reloaded_model_is_deterministic(trained):
    out_dir, cfg, _ = trained
    model, diffusion, vocab, _ = load_checkpoint(
        FAMILY, out_dir=str(out_dir), dev=torch.device("cpu")
    )
    model.eval()
    x_t = torch.randint(0, vocab.size, (3, cfg.max_len))
    t = torch.randint(1, cfg.T + 1, (3,))
    with torch.no_grad():
        a = model(x_t, t)
        b = model(x_t, t)
    assert torch.allclose(a, b)


def test_checkpoint_config_matches(trained):
    out_dir, cfg, _ = trained
    model, diffusion, vocab, meta = load_checkpoint(
        FAMILY, out_dir=str(out_dir), dev=torch.device("cpu")
    )
    assert diffusion.T == cfg.T
    assert diffusion.mask_idx == vocab.mask_idx
    assert diffusion.pad_idx == vocab.pad_idx
    assert model.max_len == cfg.max_len
    assert model.vocab_size == vocab.size
