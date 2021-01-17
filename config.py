# if True then do the training, if False do not train the model
do_training = False

# select what you want to predict - age, gender or race
# type = 'age'
# type = 'gender'
type = 'race'

n = 128

path = "data/UTKFace"

gender_classes = {
    0: 'Male',
    1: 'Female'
}

race_classes = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others'
}

num_classes = {
    'age': 6,
    'gender': 2,
    'race': 5
}

age_classes = {
    0: 'Children (1-15)',
    1: 'Youth (15-30)',
    2: 'Adults (30-40)',
    3: 'Middle age (40-60)',
    4: 'Old (60-80)',
    5: 'Very old (> 80)'
}