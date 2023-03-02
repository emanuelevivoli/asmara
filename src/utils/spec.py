TASKS = ['binary','trinary','multi']

CLASS_NUMBER = {
        "binary": 2,
        "trinary": 3,
        "multi": 13
    }

locations = ['outdoor', 'indoor']
prefixes = {
    'indoor': ['in'],
    'outdoor': ['']
}
positions = {
    'indoor': ['bas', 'low', 'pad'],
    'outdoor': ['']
}
info = {
    'indoor':{
        'prefix': 'in',
        'indexes': list(range(0, 4)),
        'keys': ["location", "distance_from_source", "id", "orientation"],
        'columns': ['file_name', 'id', 'category', 'name', 'orientation', 'distance_from_source', 'inclination', 'location']
    },
    'outdoor':{
        'prefix': 'out',
        'indexes': list(range(0, 2)),
        'keys': ["id", "orientation"],
        'columns': ['file_name', 'id', 'category', 'orientation', 'location', 'additional']
    }
}
params = {
    'indoor':{
        'MEDIUM_INDEX': 1,
    },
    'outdoor':{
        'MEDIUM_INDEX': 4,
    }
}