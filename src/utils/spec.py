TASKS = ['bin','tri','fine-grain']
CLASS_NUMBER = {
        "bin": 2,
        "tri": 3,
        "fine-grain": 5
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
        'indexes': list(range(0, 5)),
        'keys': ["location", "distance_from_source", "id", "orientation", "shape"],
        'columns': ['file_name', 'id', 'category', 'name', 'orientation', 'distance_from_source', 'inclination', 'shape', 'location']
    },
    'outdoor':{
        'prefix': 'out',
        'indexes': list(range(0, 3)),
        'keys': ["id", "orientation", "shape"],
        'columns': ['file_name', 'id', 'category', 'orientation', 'shape', 'location', 'additional']
    }
}