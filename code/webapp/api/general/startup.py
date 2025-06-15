from django.conf import settings

from api.general.utils.value_container import ValueContainer

from .data import GeneralData

data_store_: ValueContainer[GeneralData] = ValueContainer()


# Load data function
def load_data() -> None:
    """
    Load the general data in memory.
    This function is called at the startup of the application.
    """
    from .loaddata import load_data

    data_store_.set(load_data(settings.DATA_DIR))


def get_data() -> GeneralData:
    """
    Get the general data.
    This function is used to access the data in the application.
    """
    return data_store_()
