class Config:
    def __init__(self, env):

        SUPPORTED_ENVS = ['dev', 'sit', 'uat', 'prod']

        if env.lower() not in SUPPORTED_ENVS:
            raise Exception(f'{env} is not a supported environment (supported envs: {SUPPORTED_ENVS})')

        self.base_url = {
                'dev': 'http://localhost:3000/',
                'sit': 'https://harmony.sit.earthdata.nasa.gov/',
                'uat': 'https://harmony.uat.earthdata.nasa.gov/',
                'prod': 'https://harmony.earthdata.nasa.gov/'
                }[env]

        self.avnir_id = {
                'dev': 'C1233629671-ASF',
                'sit': 'C1233629671-ASF',
                'uat': 'C1233629671-ASF',
                'prod': 'C1808440897-ASF'
                }[env]

        self.grfn_id = {
                'dev': 'C1225776654-ASF',
                'sit': 'C1225776654-ASF',
                'uat': 'C1225776654-ASF',
                'prod': 'C1595422627-ASF'
                }[env]

        self.uavsar_id = {
                'dev': 'C1207038647-ASF',
                'sit': 'C1207038647-ASF',
                'uat': 'C1207038647-ASF',
                'prod': 'C1214354031-ASF'
                }[env]

        self.env_flag = {
                'dev': 'dev',
                'sit': 'sit',
                'uat': 'uat',
                'prod': 'prod'
                }[env]
