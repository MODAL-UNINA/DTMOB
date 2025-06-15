from django.conf import settings

from api.general.utils.value_container import ValueContainer

from .data import WhatIfData

_data_store: ValueContainer[WhatIfData] = ValueContainer()


# Load data function
def load_data() -> None:
    """
    Load the what-if data in memory.
    This function is called at the startup of the application.
    """
    from .loaddata import load_data

    _data_store.set(load_data(settings.DATA_DIR))


def get_data() -> WhatIfData:
    """
    Get the what-if data.
    This function is used to access the data in the application.
    """
    return _data_store()
