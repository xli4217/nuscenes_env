from future.utils import viewitems

class Configuration(object):

    def __init__(self, config={}):
        self.config = config

    def update(self, config):
        if isinstance(config, dict):
            self.config.update(config)
        else:
            self.config.update(config.config)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def keys(self):
        return self.config.keys()
        
    def get(self, key):
        '''
        key can be a string, or a list (for nested dicts)
        '''

        if isinstance(key, str):
            return self.config[key]
        elif isinstance(key, list):
            value = self.config[key[0]]
            for i in range(1, len(key)):
                value = value[key[i]]
            return value
        else:
            raise ValueError("unsupported key type")

    def set(self, key_value_list):
        '''
        for example, key_value_list = [{"key": ['agent', 'seed'], 'value': 10}, {...}]
        '''

        for key, value in key_value_list:
            key = key_value['key']
            value = key_value['value']
            if len(key) > 1:
                d = self.config[key[0]]
                for i in range(1, len(key[:-1])):
                    d = d[key[i]]
                d[key[-1]] = value
            else:
                self.config[key[0]] = value

if __name__ == "__main__":
    test_config = {
        'a': {
            'b': 1,
            'c': {
                'd': 2
            }
        },
        'e': 3
    }

    cls = Configuration(test_config)
    print(cls.config)
    print(cls.get('a'))
    print(cls.get(['a', 'c']))
    cls.set([{"key": ['a', 'b'], 'value': 19}, {'key':['e'], 'value': 13}])
    
