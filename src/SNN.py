#!/usr/bin/env python 

'''
Filename: SNN.py
Author: Jean-Sebastien Dessureault
Date created: 01/06/2016
Python version 2.7
'''
from brian2 import *
prefs.codegen.target = "cython"

import os
import sys
import rospy
import numpy as np
from std_msgs.msg import String, Float32, Float32MultiArray, Int16
import time
from lxml import etree

node = "node_SNN"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

# Retriving parameters from launcher
SNNname = rospy.get_param("~SNNname")
verbose = rospy.get_param("~verbose")
pathSNN = rospy.get_param("~path")
xml = rospy.get_param("~xml") 

# output topics
topics_motor_volts = []
topics_motor_spikes = []

# Constants
SENSORY_LAYER = 0             # input layer index
INTER_LAYER = 1
MOTOR_LAYER = 2    # Motor layer index

# Global variables
sensory_neurons = 0
inter_neurons = 0
motor_neurons = 0
synapse_delay = 0
input_drive_current = 0
tau = 0*ms
threshold_value = 0
reset_value = 0
refractory_value = 0* ms
simulation_lenght_int = 0
simulation_lenght_ms = 0* ms
realtime_limit = 50


# Global variable that receives the frames from the topic.
frames_in = []
time_frame = []
MAX_TIME_FRAME = 25
first_frame = True

# Neurons and synapses vectors    
neurons = []                # Array of neuronGroup
synapses = []               # Array of synapses
       
equation = "dv/dt = (I - v)/tau : 1 (unless refractory) I : 1"

# Test parameters validity
#useXML = True
exitSNN = False
if xml == "":
    rospy.loginfo("Must specify a XML name (XML) in the launcher.")
    exitSNN = True
if SNNname == "":
    rospy.loginfo("Must specify a SNN name (SNNname) in the launcher.")
    exitSNN = True
if pathSNN == "":
    rospy.loginfo("Must specify a path for the SNN (pathSNN) in the launcher.")
    exitSNN = True
try:
    xml_file = pathSNN + "xml/" + xml
    tree = etree.parse(xml_file)
except:
    rospy.loginfo("Error in XML file: " + xml_file)    
    exitSNN = True
# If there is at least one error in the previous test, then exit.     
if exitSNN == True:
    sys.exit(1)    

def Assing_XML():
    global tree, MOTOR_LAYER, sensory_neurons, inter_neurons, inter_layers, motor_neurons, synapse_delay, input_drive_current, tau, threshold_value, reset_value, refractory_value, simulation_lenght_int,simulation_lenght_ms, realtime_limit

    for rnd in tree.xpath("/SNN"):
        synapse_delay = rnd.get("synapse_delay")
        realtime_limit = float(rnd.get("realtime_limit"))        
        input_drive_current = float(rnd.get("input_drive_current"))
        tau = int(rnd.get("tau")) *ms
        threshold_value = rnd.get("threshold")
        reset_value = rnd.get("reset")
        refractory_value = int(rnd.get("refractory")) * ms
        simulation_lenght_int = float(rnd.get("sim_lenght"))
        simulation_lenght_ms = float(rnd.get("sim_lenght")) * ms
    
    sensory_neurons=0
    inter_neurons=0
    motor_neurons=0 
    inter_layers = 0
    for neuron in tree.xpath("/SNN/layer/neuron"):
        layer_type = neuron.getparent().get("type")
        if layer_type == "sensory":
            sensory_neurons+=1
        if layer_type == "inter":
            inter_neurons+=1
        if layer_type == "motor":
            motor_neurons+=1
           

def Display_Parameters():
    # Displaying parameters to console
    global SSNname, xml, verbose, sensory_neurons, inter_neurons, motor_neurons, inter_layers,  \
         synapse_delay, synapse_condition, input_drive_current, tau, threshold_value, refractory_value, simulation_lenght_int, pathSNN, equation, realtime_limit
    rospy.loginfo("----Parameters received from launcher OR XML file:----")
    rospy.loginfo("SNNname: " + SNNname)
    rospy.loginfo("xml: " + xml)
    rospy.loginfo("verbose: " + str(verbose))
    rospy.loginfo("Real time limit: " + str(realtime_limit))
    rospy.loginfo("sensory_neurons: " + str(sensory_neurons))
    rospy.loginfo("motor_neurons: " + str(motor_neurons))
    rospy.loginfo("inter_neurons: " + str(inter_neurons))
    rospy.loginfo("synapse_delay: " + synapse_delay)
    rospy.loginfo("input_drive_current: " + str(input_drive_current))
    rospy.loginfo("tau: " + str(tau))
    rospy.loginfo("threshold: " + str(threshold_value))
    rospy.loginfo("refractory: " + str(refractory_value))
    rospy.loginfo("simulation_lenght: " + str(simulation_lenght_int))
    rospy.loginfo("path: " + pathSNN)
    rospy.loginfo("equation: " + equation)

# Initialize input frames.
def init_frames_in():
    for x in range(0, sensory_neurons):
        frames_in[x] = 0.0

# Callback triggered when there is a new message on the topic.
def callbackReceiveMsgFromTopic(data, sensory_nb):
    #rospy.loginfo("Received in the callback: neuron: %i  dat: %s", sensory_nb, data.data)
    valeur = float(data.data) 
    if valeur != 0:
        frames_in[sensory_nb] = valeur

# Display time.  Must be called in the main SNN loop. 
def display_chrono(start, label):    
    global MAX_TIME_FRAME, first_frame, realtime_limit
    value_gtz = 0.1    
    elapsed = time.time() - start
    topic_realtime.publish(float(elapsed - realtime_limit))
    txt = "time: %.2f" % (elapsed)
    rospy.loginfo(label + " " +txt)
    if elapsed > value_gtz:
        if len(time_frame) < MAX_TIME_FRAME:
            if first_frame == False:
                time_frame.append(elapsed)
            first_frame = False
        if len(time_frame) >= MAX_TIME_FRAME:
            avg = sum(time_frame) / len(time_frame)
            rospy.loginfo("Time frame average after " +str(MAX_TIME_FRAME) + ": " + str(avg))    	
    

# Function returning the layer index
def layer_index(layer):
    global SENSORY_LAYER, MOTOR_LAYER, INTER_LAYER
    layer_index = -1
    if layer == "sensory":
        layer_index = SENSORY_LAYER
    if layer == "inter":
        layer_index = INTER_LAYER
    if layer == "motor":
        layer_index = MOTOR_LAYER
    return layer_index

# Display the message if verbose mode    
def Display(msg):
    if verbose:
        print(msg)

# Create the neurons
def Create_Neurons():
    global SENSORY_LAYER, MOTOR_LAYER, sensory_neurons, equation, threshold_value, reset_value, refractory_value, inter_neurons, motor_neurons
    Display("Creating SNN...")
    for x in range(0, sensory_neurons):
        frames_in.append(0.0)
    # Creation of the neurons 
    for layer in range(SENSORY_LAYER,MOTOR_LAYER+1): 
        # Neurons
        if layer == SENSORY_LAYER:
            neurons.append(NeuronGroup(sensory_neurons, equation, threshold=threshold_value, reset=reset_value, refractory=refractory_value, method='linear'))
            Display("Assigning SENSORY layer: " + str(layer))
        if layer == MOTOR_LAYER: 
            neurons.append(NeuronGroup(motor_neurons, equation, threshold=threshold_value, reset=reset_value, refractory=refractory_value, method='linear'))
            Display("Assigning MOTOR layer: " + str(layer))
        if layer == INTER_LAYER:
            # Each layer must have at leat on neuron. If there is 0, we change it to 1 (that won't be connected). 
            if inter_neurons == 0:
                inter_neurons = 1
            neurons.append(NeuronGroup(inter_neurons, equation, threshold=threshold_value, reset=reset_value, refractory=refractory_value, method='linear'))
            Display("Assigning INTER layer: " + str(layer))

# Create the synapses (from xml)
def Create_Synpase():
    global tree
    # APPRENDRE-PYTHON.COM/PAGE-XML-PYTHON-XPATH  
    syn_no = 0      
    for neuron in tree.xpath("/SNN/layer/neuron"):
        layer = layer_index(neuron.getparent().get("type")) 
        #print "Layer: " + str(layer)
        if layer != SENSORY_LAYER :
            layer_from = layer_index(neuron.get("layer"))
            layer_to = layer
            neuron_from = neuron.get("synapse")
            neuron_to = neuron.get("id") 
            the_weights_str = neuron.get("weight")
            the_weights = the_weights_str.split(',')
            synapses.append(Synapses(neurons[layer_from], neurons[layer_to], model='w: 1', on_pre='v += w', multisynaptic_index = 'synapse_number'))             
            str_connect = "i="+str(neuron_from)+" j="+str(neuron_to)
            print "Synapse:  From: " + str(layer_from) + " To: " + str(layer_to) + " condition: " + str_connect + " synapses: " + the_weights_str
            synapses[syn_no].connect(i=eval(neuron_from), j=eval(neuron_to))
            for i in range (0, len(the_weights)):
                #print "w: " + str(eval(the_weights[i]))
                synapses[syn_no].w[i] = eval(the_weights[i]) 
            syn_no += 1
    #sys.exit(1)    


# Simulation of SNN
def Simulation():
    # Integrtion of each component in the network. 
    Display("Integration of each component in the network.")
    stateMotor = StateMonitor(neurons[MOTOR_LAYER], 'v', record=True)
    spikeMonitor = SpikeMonitor(neurons[MOTOR_LAYER])
    net = Network(collect())
    net.add(neurons)
    net.add(synapses)
    
    Display("Restoring previously learned SNN...")
    net.store()

    init_frames_in()
    
    # Main loop.  Inifite if RUN mode.   Quit after X iteration if LEARNING mode. 
    theExit = False
    while True: 
        cycle_msg = "\n"
        # Start the cycle and the timer.
        start = time.time()
        time.clock()
        

        # Restore initial SNN and monitors
        net.restore()
        
        # When the callback function has received all the input neurons, assign those neurons to the input layer. 
        frames_assignation = frames_in
        #rospy.loginfo("Assigned sensories: " + str(frames_assignation))

        # Assing sensory neurons from frames              
        for k in range(0,sensory_neurons): 
            neurons[SENSORY_LAYER].v[k] = frames_assignation[k]   # Only v of the sensory neurons
            neurons[SENSORY_LAYER].I[k] = input_drive_current 
            #rospy.loginfo("neuron " + str(k) + " v. " + str(neurons[SENSORY_LAYER].v[k])) 
            cycle_msg += "neuron " + str(k) + " v. " + str(neurons[SENSORY_LAYER].v[k]) + "\n"
        init_frames_in()
        # Simulation execution
        net.run(simulation_lenght_ms)
                      
        # Publish on the output topic
        for y in range(0, motor_neurons):
            #rospy.loginfo("Values to publish for neuron " + str(y) + " : " + str(len(stateMotor.v[y])))
            #rospy.loginfo("Neuron: " + str(y) + " Spikes: " + str(spikeMonitor.num_spikes))
            # publish volts
            voltsToPublish = Float32MultiArray()
            voltsToPublish.data = stateMotor.v[y]
            topics_motor_volts[y].publish(voltsToPublish) 
            # publish spikes    
            nb_spikes = sum(spikeNo == y for spikeNo in spikeMonitor.i)
            topics_motor_spikes[y].publish(nb_spikes)
            cycle_msg += "Neuron: " + str(y) + " Spikes: " + str(nb_spikes) + "\n"
        topic_simulation_lenght.publish(simulation_lenght_int)
        #rospy.loginfo("Transmitted voltage values: "  + str(len(stateMotor.v[0])))

        # End of the cycle
        os.system("clear")
        display_chrono(start, "LAST CYCLE DATA:")    
        Display(cycle_msg)
                

# MAIN SSN Functions

# SNN 
def SNN():
    rospy.loginfo("Starting SNN process...")
    start_scope()
    Create_Neurons()
    Create_Synpase()
    Simulation()


Assing_XML()
Display_Parameters()
# Declaring the topics 
Display("Subscribe to the callbacks (input neurons)...")
for k in range(0, sensory_neurons):
    rospy.Subscriber("/"+SNNname+"_"+str(k+1)+"_snn_in", String, callbackReceiveMsgFromTopic, k)
    
for k in range(0, motor_neurons):
    topics_motor_volts.append(rospy.Publisher('motor_volts_'+SNNname+str(k+1), Float32MultiArray, queue_size=1))
    topics_motor_spikes.append(rospy.Publisher('motor_spikes_'+SNNname+str(k+1), Float32, queue_size=1))
topic_simulation_lenght = rospy.Publisher('simulation_lenght_'+SNNname, Int16, queue_size=1)
topic_realtime = rospy.Publisher('realtime_'+SNNname, Float32, queue_size=1)

SNN()

