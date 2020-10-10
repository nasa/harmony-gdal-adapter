from pytest import fixture
from config import Config

# Set up the ability to run pytest with --env <environment> option
def pytest_addoption(parser):
    parser.addoption(
            "--env",
            action="store",
            help="Environment to run tests against (dev, sit, uat, prod)"
            )

@fixture(scope='session')
def env(request):
    return request.config.getoption("--env")

@fixture(scope='session')
def harmony_url_config(env):
    cfg = Config(env)
    return cfg
