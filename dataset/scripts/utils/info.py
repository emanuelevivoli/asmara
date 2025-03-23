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
