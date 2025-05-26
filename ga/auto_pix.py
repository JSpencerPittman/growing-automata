from ga import Automata, AutomataConfig, OptimizerConfig
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np
import torch


@dataclass
class PixelArtAutomataConfig(object):
    state_size: int
    optimizer: OptimizerConfig
    gen_rnds: int


class PixelArtAutomata(Automata):
    def __init__(self, config: PixelArtAutomataConfig, img_path: Path):
        def load_pixart(path: Path) -> np.ndarray:
            assert path.exists() and path.is_file()
            img = Image.open(path)
            assert img.mode == "P"
            return np.array(img.convert("RGB"))

        self.tgt_pixart = load_pixart(img_path)
        config_super = AutomataConfig(
            self.tgt_pixart.shape[:2],
            config.state_size,
            3,
            256,
            config.optimizer,
            config.gen_rnds,
        )
        expected = torch.from_numpy(self.tgt_pixart).float()

        super().__init__(config_super, expected)

    def save_output_to_img(self, path: Path):
        with torch.no_grad():
            output = self.output()
        img = output.detach().numpy().astype(np.uint8)
        print(img)
        Image.fromarray(img).save(path)


if __name__ == "__main__":
    optim_config = OptimizerConfig(torch.optim.Adam, {"lr": 1e-4})
    automata_config = PixelArtAutomataConfig(1, optim_config, 10)
    automata = PixelArtAutomata(automata_config, Path("samples/minecraft.png"))

    automata.train(10)
    automata.eval()
    automata.save_output_to_img(Path("output.png"))
