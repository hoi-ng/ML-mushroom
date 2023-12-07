from flask import Flask,request,jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd

model = tf.keras.models.load_model('model.h5')

app = Flask(__name__)
@app.route('/', methods=["GET","POST"])
def index():
    print("Hello world")
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    print("Received POST request")
    if request.is_json:
        data = request.json
    else:
        data = dict(request.values)
    cap_shape = data['cap_shape']
    cap_surface = data['cap_surface']
    cap_color = data['cap_color']
    bruises = data['bruises']
    odor = data['odor']
    gill_attachment = data['gill_attachment']
    gill_spacing = data['gill_spacing']
    gill_size = data['gill_size']
    gill_color = data['gill_color']
    stalk_shape = data['stalk_shape']
    stalk_surface_above_ring = data['stalk_surface_above_ring']
    stalk_surface_below_ring = data['stalk_surface_below_ring']
    stalk_color_above_ring = data['stalk_color_above_ring']
    stalk_color_below_ring = data['stalk_color_below_ring']
    veil_color = data['veil_color']
    ring_number = data['ring_number']
    ring_type = data['ring_type']
    spore_print_color = data['spore_print_color']
    population = data['population']
    habitat = data['habitat']
    
    cap_shape_encoder = LabelEncoder()
    cap_shape_encoder.classes_ = np.load('cap_shape_encoder.npy', allow_pickle=True)
    cap_surface_encoder = LabelEncoder()
    cap_surface_encoder.classes_ = np.load('cap_surface_encoder.npy', allow_pickle=True)
    cap_color_encoder = LabelEncoder()
    cap_color_encoder.classes_ = np.load('cap_color_encoder.npy', allow_pickle=True)
    bruises_encoder = LabelEncoder()
    bruises_encoder.classes_ = np.load('bruises_encoder.npy', allow_pickle=True)
    odor_encoder = LabelEncoder()
    odor_encoder.classes_ = np.load('odor_encoder.npy', allow_pickle=True)
    gill_attachment_encoder = LabelEncoder()
    gill_attachment_encoder.classes_ = np.load('gill_attachment_encoder.npy', allow_pickle=True)
    gill_spacing_encoder = LabelEncoder()
    gill_spacing_encoder.classes_ = np.load('gill_spacing_encoder.npy', allow_pickle=True)
    gill_size_encoder = LabelEncoder()
    gill_size_encoder.classes_ = np.load('gill_size_encoder.npy', allow_pickle=True)
    gill_color_encoder = LabelEncoder()
    gill_color_encoder.classes_ = np.load('gill_color_encoder.npy', allow_pickle=True)
    stalk_shape_encoder = LabelEncoder()
    stalk_shape_encoder.classes_ = np.load('stalk_shape_encoder.npy', allow_pickle=True)
    stalk_surface_above_ring_encoder = LabelEncoder()
    stalk_surface_above_ring_encoder.classes_ = np.load('stalk_surface_above_ring_encoder.npy', allow_pickle=True)
    stalk_surface_below_ring_encoder = LabelEncoder()
    stalk_surface_below_ring_encoder.classes_ = np.load('stalk_surface_below_ring_encoder.npy', allow_pickle=True)
    stalk_color_above_ring_encoder = LabelEncoder()
    stalk_color_above_ring_encoder.classes_ = np.load('stalk_color_above_ring_encoder.npy', allow_pickle=True)
    stalk_color_below_ring_encoder = LabelEncoder()
    stalk_color_below_ring_encoder.classes_= np.load('stalk_color_below_ring_encoder.npy', allow_pickle=True)
    veil_color_encoder = LabelEncoder()
    veil_color_encoder.classes_ = np.load('veil_color_encoder.npy', allow_pickle=True)
    ring_number_encoder = LabelEncoder()
    ring_number_encoder.classes_ = np.load('ring_number_encoder.npy', allow_pickle=True)
    ring_type_encoder = LabelEncoder()
    ring_type_encoder.classes_ = np.load('ring_type_encoder.npy', allow_pickle=True)
    spore_print_color_encoder = LabelEncoder()
    spore_print_color_encoder.classes_ = np.load('spore_print_color_encoder.npy', allow_pickle=True)
    population_encoder = LabelEncoder()
    population_encoder.classes_ = np.load('population_encoder.npy', allow_pickle=True)
    habitat_encoder = LabelEncoder()
    habitat_encoder.classes_ = np.load('habitat_encoder.npy', allow_pickle=True)

    cap_shape = cap_shape_encoder.transform([cap_shape])[0]
    cap_surface = cap_surface_encoder.transform([cap_surface])[0]
    cap_color= cap_color_encoder.transform([cap_color])[0]
    bruises = bruises_encoder.transform([bruises])[0]
    odor = odor_encoder.transform([odor])[0]
    gill_attachment = gill_attachment_encoder.transform([gill_attachment])[0]
    gill_spacing = gill_spacing_encoder.transform([gill_spacing])[0]
    gill_size = gill_size_encoder.transform([gill_size])[0]
    gill_color = gill_color_encoder.transform([gill_color])[0]
    stalk_shape = stalk_shape_encoder.transform([stalk_shape])[0]
    stalk_surface_above_ring = stalk_surface_above_ring_encoder.transform([stalk_surface_above_ring])[0]
    stalk_surface_below_ring = stalk_surface_below_ring_encoder.transform([stalk_surface_below_ring])[0]
    stalk_color_above_ring= stalk_color_above_ring_encoder.transform([stalk_color_above_ring])[0]
    stalk_color_below_ring = stalk_color_below_ring_encoder.transform([stalk_color_below_ring])[0]
    veil_color = veil_color_encoder.transform([veil_color])[0]
    ring_number = ring_number_encoder.transform([ring_number])[0]
    ring_type = ring_type_encoder.transform([ring_type])[0]
    spore_print_color = spore_print_color_encoder.transform([spore_print_color])[0]
    population = population_encoder.transform([population])[0]
    habitat = habitat_encoder.transform([habitat])[0]

    input_query = np.array([[cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat]])
    #print("input_query",input_query)
    
    #print(f"INPUT QUERY {input_query}")
    result = model.predict(input_query)[0]

    #result = {'cap_shape':cap_shape, 'cap_surface':cap_surface, 'cap_color':cap_color, 'bruises':bruises, 'odor':odor, 'gill_attachment':gill_attachment, 'gill_spacing':gill_spacing, 'gill_size':gill_size, 'gill_color':gill_color, 'stalk_shape':stalk_shape,'stalk_surface_above_ring':stalk_surface_above_ring,'stalk_surface_below_ring':stalk_surface_below_ring,'stalk_color_above_ring':stalk_color_above_ring,'stalk_color_below_ring':stalk_color_below_ring, 'veil_color':veil_color, 'ring_number':ring_number,'ring_type':ring_type, 'spore_print_color':spore_print_color, 'population':population, 'habitat':habitat}

    return jsonify({'RESULT':str(result)})

if __name__ == '__main__':
    app.run(debug=True)
