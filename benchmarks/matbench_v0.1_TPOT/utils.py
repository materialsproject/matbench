import pickle
import re
from bisect import bisect_left


class LoadExisting:
    
    def __init__(self):
        pass
    
    def cleaning(self, inputs):
        X = []
        possible_elements = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'N', 'Nb', 'Co', 'W', 'Al', 'Ti'] 
        for sample in inputs:
            values=re.findall(r"(?:\d*\.\d+|\d+)",sample)
            actual_elements=re.findall(r"(\w+?)(?:\d*\.\d+|\d+)",sample)
            # Assert that the lengths of actual_elements and values are the same
            assert len(actual_elements) == len(values)

            # Sort actual_elements and values alphabetically by actual_elements
            actual_elements, values = zip(*sorted(zip(actual_elements, values)))

            # Sort possible_elements alphabetically
            possible_elements.sort()

            # Iterate through each element in possible_elements
            for element in possible_elements:
                # If the element is not present in actual_elements
                if element not in actual_elements:
                    # Find the index where the element should be inserted alphabetically
                    index = bisect_left(actual_elements, element)

                    # Insert the element at the corresponding index in actual_elements
                    actual_elements = actual_elements[:index] + (element,) + actual_elements[index:]

                    # Insert a zero at the same position in the values list
                    values = values[:index] + (0,) + values[index:]
            X.append(values)
        return X
