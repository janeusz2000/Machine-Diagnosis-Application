import numpy as np
from dataclasses import dataclass


@dataclass
class BearingFrequencies:
    inner_ring_frequency: float
    outer_ring_Frequency: float
    rolling_element_set_and_cage_frequency: float
    rolling_element_about_its_axis_frequency: float
    point_on_inner_ring_Frequency: float
    point_on_outer_ring_Frequency: float
    rolling_element_frequency: float

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __repr__(self) -> str:
        message = ""
        for attr, value in self:
            message += f"{str(attr)} : {value}\n"
        return f"Bearing Frequency Ratios: \n{message}"

    def __str__(self) -> str:
        return self.__repr__()

    def getCharacteristicFrequencies(self, inputFrequency: float):
        outputFrequencies = {}
        for attr, value in self:
            outputFrequencies[attr] = inputFrequency * value
        return outputFrequencies


class Bearing6306Frequencies(BearingFrequencies):
    # numbers obtained from https://www.skfbearingselect.com/
    # with single rolling bearing 6306 selected for input frequency 1Hz.
    def __init__(self):
        super().__init__(inner_ring_frequency=0.016666,
                         outer_ring_Frequency=0,
                         rolling_element_set_and_cage_frequency=0.006362,
                         rolling_element_about_its_axis_frequency=0.03325,
                         point_on_inner_ring_Frequency=0.08244,
                         point_on_outer_ring_Frequency=0.050894,
                         rolling_element_frequency=0.0665)
