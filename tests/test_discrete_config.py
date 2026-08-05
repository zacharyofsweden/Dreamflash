"""Unit tests for discrete GPU memory configuration generator."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "tools" / "roofline"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "discrete_config"))

from config_generator import export_c_header, generate_discrete_config
from model import FLASH


class TestDiscreteConfig(unittest.TestCase):

    def test_discrete_config_calculation(self) -> None:
        """Test calculation of discrete VRAM and Host RAM capacity."""
        cfg = generate_discrete_config(vram_gb=12.0, ram_gb=16.0, ctx_tokens=100_000)
        self.assertGreater(cfg["vram_expert_count"], 0)
        self.assertGreater(cfg["host_expert_count"], 0)
        self.assertEqual(cfg["vram_expert_count"] + cfg["host_expert_count"], cfg["total_cache_experts"])
        self.assertLess(cfg["total_cache_experts"], FLASH.total_routed_experts)

    def test_c_header_export(self) -> None:
        """Test exporting C header file."""
        cfg = generate_discrete_config(vram_gb=12.0, ram_gb=16.0, ctx_tokens=100_000)
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".h") as tmp:
            tmp_path = tmp.name

        try:
            export_c_header(cfg, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("#define DS4_CONFIG_VRAM_EXPERT_CAPACITY", content)
            self.assertIn(str(cfg["vram_expert_count"]), content)
            self.assertIn(str(cfg["host_expert_count"]), content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
