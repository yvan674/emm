import logging
import pandas as pd

from description import Description


class Subgroup:

    def __init__(self, data: pd.DataFrame, description: Description):
        """Implementation of a subgroup.

        Basically a pandas DataFrame with additional description. Also holds
        other info such as score, target, and coverage for later calculations.
        """
        self.data = data
        self.description = description
        self.score = None
        self.target = None
        self.coverage = None

    def decrypt_description(self, translation):
        self.description.decrypt(translation)

    @property
    def size(self):
        return len(self.data)

    def print(self):
        logging.debug(f"{str(self.description)} {self.score} ({self.size})")
