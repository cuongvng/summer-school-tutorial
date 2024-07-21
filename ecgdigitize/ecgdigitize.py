from ecgdigitize import visualization
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ecgdigitize.image import ColorImage
from . import common
from .grid import detection as grid_detection
from .grid import extraction as grid_extraction
from .signal import detection as signal_detection
from .signal.extraction import viterbi
from . import vision


def estimateRotationAngle(image: ColorImage, houghThresholdFraction: float = 0.5) -> Optional[float]:
    binaryImage = grid_detection.thresholdApproach(image)

    houghThreshold = int(image.width * houghThresholdFraction)
    lines = vision.houghLines(binaryImage, houghThreshold)

    # <- DEBUG ->
    # visualization.displayImage(visualization.overlayLines(lines, image))

    angles = common.mapList(lines, vision.houghLineToAngle)
    offsets = common.mapList(angles, lambda angle: angle % 90)
    candidates = common.filterList(offsets, lambda offset: abs(offset) < 30)

    if len(candidates) > 1:
        estimatedAngle = common.mean(candidates)
        return estimatedAngle
    else:
        return None
    

def digitizeSignal(image: ColorImage) -> Union[np.ndarray, common.Failure]:
    
    # First, convert color image to binary image where signal pixels are turned on (1) and other are off (0)
    binary = signal_detection.adaptive(image)

    # Second, analyze the binary image to produce a signal
    signal = viterbi.extractSignal(binary)

    return signal


def digitizeGrid(image: ColorImage) -> Union[float, common.Failure]:  
    # Returns size of grid in pixels

    # First, convert color image to binary image where grid pixels are turned on (1) and all others are off (0)
    binary = grid_detection.allDarkPixels(image)

    # Second, analyze the binary image to estimate the grid spacing (period)
    gridPeriod = grid_extraction.estimateFrequencyViaAutocorrelation(binary.data)

    return gridPeriod

