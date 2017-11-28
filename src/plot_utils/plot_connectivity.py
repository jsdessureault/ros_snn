#!/usr/bin/env python 

'''
Filename: SNN.py
Author: Jean-Sebastien Dessureault
Date created: 01/06/2016
Python version 2.7
'''
from brian2 import *
prefs.codegen.target = "cython"

import rospy
import numpy as np
import time
import string
from lxml import etree
import matplotlib.pyplot as plt

# Registering node to ROS
node = "node_connectivity_plot"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

# Retriving parameters from launcher
verbose = rospy.get_param("~verbose")
pathSNN = rospy.get_param("~path")
xml = rospy.get_param("~xml")
SNNname = rospy.get_param("~SNNname")

sensory_neurons = rospy.get_param("~sensory_neurons")
motor_neurons = rospy.get_param("~motor_neurons")
inter_neurons = rospy.get_param("~inter_neurons")
inter_layers = rospy.get_param("~inter_layers")
synapse_weight = str(rospy.get_param("~synapse_weight"))
synapse_delay = str(rospy.get_param("~synapse_delay"))
synapse_condition = rospy.get_param("~synapse_condition")
input_drive_current = rospy.get_param("~input_drive_current")
tau = rospy.get_param("~tau") * ms
threshold_value = rospy.get_param("~threshold")
refractory_value = rospy.get_param("~refractory") * ms
reset_value = rospy.get_param("~reset")
simulation_lenght_ms = rospy.get_param("~simulation_lenght") * ms
simulation_lenght_int = rospy.get_param("~simulation_lenght") 

# Displaying parameters to console
rospy.loginfo("----Parameters received from launcher OR XML file:----")
rospy.loginfo("SNNname: " + SNNname)
rospy.loginfo("xml: " + xml)
rospy.loginfo("verbose: " + str(verbose))
rospy.loginfo("sensory_neurons: " + str(sensory_neurons))
rospy.loginfo("motor_neurons: " + str(motor_neurons))
rospy.loginfo("inter_neurons: " + str(inter_neurons))
rospy.loginfo("inter_layers: " + str(inter_layers))
rospy.loginfo("synapse_weight: " + synapse_weight)
rospy.loginfo("synapse_delay: " + synapse_delay)
rospy.loginfo("synapse_condition: " + synapse_delay)
rospy.loginfo("input_drive_current: " + str(input_drive_current))
rospy.loginfo("tau: " + str(tau))
rospy.loginfo("threshold: " + str(threshold_value))
rospy.loginfo("refractory: " + str(refractory_value))
rospy.loginfo("simulation_lenght: " + str(simulation_lenght_int))
rospy.loginfo("path: " + pathSNN)

# Registering node to ROS
#rospy.init_node('node_plot_connectivity_'+SNNname, anonymous=True)
#rospy.loginfo("SNN - Plot connectivity - " + SNNname)

# Function returning the layer index
def layer_index(layer):
    global SENSORY_LAYER, MOTOR_LAYER
    INTER_LAYER = 1
    layer_index = -1
    if layer == "sensory":
        layer_index = SENSORY_LAYER
    if layer == "inter":
        layer_index = INTER_LAYER
    if layer == "motor":
        layer_index = MOTOR_LAYER
    return layer_index

def nb_neuron(layer_no):
    global sensory_neurons, inter_neurons, motor_neurons
    if layer_no == 0:
        return sensory_neurons
    if layer_no == 1:
        return inter_neurons
    if layer_no == 2:
        return motor_neurons
    return -1

# Define neurons positions in the graph
neurons_x = []
neurons_y = []    
neurons_names = []
synapse_layer_from_to = []
synapse_neuron_from_to = []  
synapse_weight = []      
     
if xml != "":
    xml_file = pathSNN + "xml/" + xml
    rospy.loginfo("xml:" + xml_file)
    try:

        tree = etree.parse(xml_file)
        # Get the SNN variables.  Will be placed in a legend beside the graph. 
        for rnd in tree.xpath("/SNN"):
            print "Look for variables"
            name = rnd.get("name")
            #synapse_weight = rnd.get("synapse_weight")
            #synapse_delay = rnd.get("synapse_delay")
            #input_drive_current = float(rnd.get("input_drive_current"))
            #tau = int(rnd.get("tau")) *ms
            #threshold_value = rnd.get("threshold")
            #reset_value = rnd.get("reset")
            #refractory_value = int(rnd.get("refractory")) * ms
            #simulation_lenght_int = float(rnd.get("sim_lenght"))
            #simulation_lenght_ms = float(rnd.get("sim_lenght")) * ms
            
        # Defining number of each layer
        sensory_neurons = 0
        inter_neurons = 0
        motor_neurons = 0 
        inter_layers = 0
        for neuron in tree.xpath("/SNN/layer/neuron"):
            layer_type = neuron.getparent().get("type")
            if layer_type == "sensory":
                sensory_neurons+=1
            if layer_type == "inter":
                inter_neurons+=1
                inter_layers = 1
            if layer_type == "motor":
                motor_neurons+=1
            neurons_names.append(neuron.text)
        
        # Constants
        SENSORY_LAYER = 0             # input layer index
        MOTOR_LAYER = inter_layers + 2 - 1    # Motor layer index:  inter layer +  1 sensory layer + 1 motor layer (- 1 because the index starts at 0).
        
        rospy.loginfo("Sensory neurons " + str(sensory_neurons))
        rospy.loginfo("Inter layers " + str(inter_layers))
        rospy.loginfo("Inter neurons " + str(inter_neurons))
        rospy.loginfo("Motor neurons " + str(motor_neurons))
            
        for layer_no in range(SENSORY_LAYER,MOTOR_LAYER+2):
            print "Processing layer #" + str(layer_no)
            for neuron_no in range(0, nb_neuron(layer_no)):
                #print "...Processing neuron #" + str(neuron_no) + " of layer #" + str(layer_no)
                neurons_y.append(layer_no)
                neurons_x.append(neuron_no)
                #print "Neuron position: " + str(layer_no) + ", " + str(neuron_no)

        # Define synapses links between the neurons
        nb_synapses = 0
        for neuron in tree.xpath("/SNN/layer/neuron"):
            layer_type = neuron.getparent().get("type")
            neuron_id = neuron.get("id")
            synapse_id = neuron.get("synapse")         
            if synapse_id != None:
                synapse_layer = neuron.get("layer")
                str_tmp = neuron.get("weight")
                synapse_weight_tmp = str_tmp.split(",")
                i = 0
                for syn in synapse_id.split(","):
                    #print "...Processing synapse #" + str(nb_synapses)
                    synapse_layer_from_to.append((layer_index(layer_type)-1, layer_index(layer_type)))
                    synapse_neuron_from_to.append((int(syn), int(neuron_id)))
                    synapse_weight.append(synapse_weight_tmp[i])
                    #print "nb_synapses: " + str(nb_synapses) + " W: " + str(synapse_weight[nb_synapses]) 
                    i += 1
                    nb_synapses += 1
            
    except:
        rospy.loginfo("Error parsing XML file: " + xml_file)    
        rospy.loginfo("Error: " + str(sys.exc_info()[0]))
        sys.exit(1)

plt.scatter(neurons_y, neurons_x, c='black', s=512, marker='o')
dy_annotation = 0.25
for i, label in enumerate(neurons_names):
    plt.annotate(label, (neurons_y[i], neurons_x[i]), xytext=(neurons_y[i], neurons_x[i]+dy_annotation))

#print synapse_layer_from_to
#print synapse_neuron_from_to

for i in range(0, nb_synapses):
    plt.plot(synapse_layer_from_to[i], synapse_neuron_from_to[i], linestyle='solid', color='black')
    de_x = synapse_layer_from_to[i][0]
    de_y = synapse_neuron_from_to[i][0]
    a_x = synapse_layer_from_to[i][1]
    a_y = synapse_neuron_from_to[i][1]
    #print "De: x: " + str(de_x) + " y: " + str(de_y) + "  A: x: " + str(a_x) + " y: " + str(a_y)
    px =  (de_x + a_x) / 2.0
    py =  (de_y + a_y) / 2.0
    #print str(i) + " y: " +str(py) + " x: " + str(px) + " w: " + str(synapse_weight[i])
    plt.annotate('W: ' +str(synapse_weight[i]), xy=(px,py) , xytext=(px,py)) 

plt.title("Connectivity - " + name)
#plt.suptitle( 
#    "Input drive current: " + str(input_drive_current) + 
#    "Threshold - " + str(threshold_value) +
#    "Tau - " + str(tau) +
#    "Refractory - " + str(refractory_value) +
#    "Reset - " + str(reset_value) +
#    "Simulation lenght - " + str(simulation_lenght_int))

# Remove ticks (axis scaling) and axis
plt.xticks([],[])
plt.yticks([],[])
plt.axis('off')
# Show the graph
plt.show()



