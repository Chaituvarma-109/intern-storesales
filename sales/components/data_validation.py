import pandas as pd
import pandera as pa

from sales.constants import ITEM_FAT_CONTENT, OUTLET_SIZE, OUTLET_LOCATION_TYPE, OUTLET_TYPE
from sales.config.configuration import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.validation_config = config

    def get_train_test_file_path(self):
        train_data_ = pd.read_csv(self.validation_config.train_data_path)
        test_data_ = pd.read_csv(self.validation_config.test_data_path)
        return train_data_, test_data_


class Schema(pa.DataFrameModel):
    Item_Identifier: object
    Item_Weight: float = pa.Field(nullable=True)
    Item_Fat_Content: object = pa.Field(isin=ITEM_FAT_CONTENT)
    Item_Visibility: float
    Item_Type: object
    Item_MRP: float
    Outlet_Identifier: object
    Outlet_Establishment_Year: int
    Outlet_Size: object = pa.Field(isin=OUTLET_SIZE, nullable=True)
    Outlet_Location_Type: object = pa.Field(isin=OUTLET_LOCATION_TYPE)
    Outlet_Type: object = pa.Field(isin=OUTLET_TYPE)
    Item_Outlet_Sales: float
