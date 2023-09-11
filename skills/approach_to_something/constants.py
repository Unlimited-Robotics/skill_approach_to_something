NO_TARGET_TIMEOUT_LONG = 3.0
NO_TARGET_TIMEOUT_SHORT= 0.4

HANDLER_NAMES = {
        'ObjectsDetectorHandler': 'object_name',
        'TagsDetectorHandler': 'tag_id',
        'FacesDetectorHandler': None,
        'FacesRecognizerHandler': 'recognition_name',
        'ObjectsSegmentatorHandler': 'object_name',
}

MAXIMUM_VALID_TIME_DETECTION = 0.2 ############## Checkear
MINIMUM_GET_TARGET_TIMEOUT= 0.4
MAX_INITIAL_ANGLE_ERROR = 60.0
MAX_MISALIGNMENT = 1.0
MIN_CORRECTION_DISTANCE = 0.5

ERROR_INVALID_ANGLE = (1, f'Invalid angle, must be between -180 and 180')
ERROR_INVALID_PREDICTOR = (2, f'Invalid predictor')
ERROR_IDENTIFIER_NOT_DEFINED = (3, f'Identifier must be defined')
ERROR_NOT_LOCALIZED = (4, f'The robot must be localized')
ERROR_INITIAL_ANGLE_TOO_FAR = 5
ERROR_NO_TARGET_FOUND = (6, f'Not target found after {NO_TARGET_TIMEOUT_LONG}')
ERROR_TOO_DISALIGNED = 7
ERROR_TOO_CLOSE_TO_TARGET = 8
ERROR_TOO_FAR_TO_TARGET = 9 
ERROR_MOVING = 10 ############## Checkear

