from django.conf import settings

from api.general.utils.value_container import ValueContainer

from .data import StatsData

_data_store: ValueContainer[StatsData] = ValueContainer()


# Load data function
def load_data() -> None:
    """
    Load the stats data in memory.
    This function is called at the startup of the application.
    """
    from .loaddata import load_data

    _data_store.set(load_data(settings.DATA_DIR))


def get_data() -> StatsData:
    """
    Get the stats data.
    This function is used to access the data in the application.
    """
    return _data_store()
