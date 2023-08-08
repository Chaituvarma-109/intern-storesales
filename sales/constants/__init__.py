from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")

CATEGORICAL_COLS = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
                    'Outlet_Location_Type', 'Outlet_Type']

NUMERICAL_COLS = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']

TARGET_COL = 'Item_Outlet_Sales'

ITEM_FAT_CONTENT = ['Low Fat', 'Regular', 'LF', 'reg', 'low fat']
OUTLET_SIZE = ['Medium', 'High', 'Small']
OUTLET_LOCATION_TYPE = ['Tier 1', 'Tier 3', 'Tier 2']
OUTLET_TYPE = ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3']
