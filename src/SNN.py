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


# Registering node to ROS
#rospy.init_node('node_SNN_'+SNNname, anonymous=True)
#rospy.loginfo("SNN - Spiking Neural Network - " + SNNname)

if xml != "":
    xml_file = pathSNN + "xml/" + xml
    rospy.loginfo("xml:" + xml_file)
    try:
        tree = etree.parse(xml_file)
        for rnd in tree.xpath("/SNN"):
            synapse_weight = rnd.get("synapse_weight")
            synapse_delay = rnd.get("synapse_delay")
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
        inter_layers = 1
        for neuron in tree.xpath("~layer/neuron"):
            layer_type = neuron.getparent().get("type")
            if layer_type == "sensory":
                sensory_neurons+=1
            if layer_type == "inter":
                inter_neurons+=1
            if layer_type == "motor":
                motor_neurons+=1

    except:
        rospy.loginfo("Error in XML file: " + xml_file)    
        rospy.loginfo("Error: " + str(sys.exc_info()[0]))
        sys.exit(1)

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

# Filenames and path where the trained SNN and pickle files will be saved. 
initFile = SNNname + "_initialized"
learnedFile = SNNname + "_learned"

# output topics
topics_motor_volts = []
topics_motor_spikes = []

# Constants
SENSORY_LAYER = 0             # input layer index
MOTOR_LAYER = inter_layers + 2 - 1    # Motor layer index:  inter layer +  1 sensory layer + 1 motor layer (- 1 because the index starts at 0).

# Global variable that receives the frames from the topic.
frames_in = [sensory_neurons]
for x in range(0, sensory_neurons):
    frames_in.append(0.0)

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
    elapsed = time.time() - start
    txt = "time: %.2f" % (elapsed)
    rospy.loginfo(label + " " +txt)

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
    

# Main SNN function.    
def SNN():
        
    start_scope()
    
    if verbose:
        rospy.loginfo("Creating SNN...")
    
    # SNN Creation    
    neurons = []                # Array of neuronGroup
    synapses = []               # Array of synapses
           
    # Creation of the equation
    # LI&F equation p.110 Brian2.pdf
    #equation = '''
    #dv/dt = (v0 - v)/tau : 1 (unless refractory)
    #v0 : 1'''

    #equation = "dv/dt = (I - v)/tau : 1 (unless refractory) I = " + input_drive_current + " : 1"
    equation = "dv/dt = (I - v)/tau : 1 (unless refractory) I : 1"

    if verbose: 
        rospy.loginfo("Equation: " + equation)
    
    # Creation of the neurons and synapses structures
    for layer in range(SENSORY_LAYER,MOTOR_LAYER+1): 
        # Neurons
        if layer == SENSORY_LAYER:
            neurons.append(NeuronGroup(sensory_neurons, equation, threshold=threshold_value, reset=reset_value, refractory=refractory_value, method='linear'))
            if verbose: 
                rospy.loginfo("Assigning SENSORY layer: " + str(layer))
        if layer == MOTOR_LAYER: 
            neurons.append(NeuronGroup(motor_neurons, equation, threshold=threshold_value, reset=reset_value, refractory=refractory_value, method='linear'))
            if verbose: 
                rospy.loginfo("Assigning MOTOR layer: " + str(layer))
        if layer < MOTOR_LAYER and layer > SENSORY_LAYER:
            neurons.append(NeuronGroup(inter_neurons, equation, threshold=threshold_value, reset=reset_value, refractory=refractory_value, method='linear'))
            if verbose: 
                rospy.loginfo("Assigning INTER layer: " + str(layer))
        # Synapses
        if layer > SENSORY_LAYER:
            if xml == "":
                postsynaptic = "v_post += " + synapse_weight    # Synapse weight 
                synapses.append(Synapses(neurons[layer-1], neurons[layer],  on_pre=postsynaptic, multisynaptic_index = 'synapse_number'))  # Fix synaptic weight to the parameter value. 
                #synapses.append(Synapses(neurons[layer-1], neurons[layer], ""))  # Synaptic weight has been change in the training mode

                # Connextion type between layers.
                if synapse_condition != "":
                    synapses[layer-1].connect(condition=synapse_condition) 
                else:
                    synapses[layer-1].connect() 
                # A delay is defined to better visualize graphics (no line overlapping).      
                synapses[layer-1].delay = 'synapse_number*'+synapse_delay+'*ms'
                if verbose: 
                    rospy.loginfo("Assigning SYNAPSES between layer: " + str(layer-1) + " and layer " + str(layer))

    if xml != "":
        global tree

        # APPRENDRE-PYTHON.COM/PAGE-XML-PYTHON-XPATH  
        syn_no = 0      
        for neuron in tree.xpath("~layer/neuron"):
            layer = layer_index(neuron.getparent().get("type")) 
            #print "Layer: " + str(layer)
            if layer != SENSORY_LAYER :
                layer_from = layer - 1
                layer_to = layer
                the_from = neuron.get("synapse")
                the_to = neuron.get("id") 
                the_weights_str = neuron.get("weight")
                the_weights = the_weights_str.split(',')
                
                synapses.append(Synapses(neurons[layer_from], neurons[layer_to], model='w: 1', on_pre='v += w', multisynaptic_index = 'synapse_number')) 
                str_connect = "i="+str(the_from)+" j="+str(the_to)
                print "Synapse:  From: " + str(layer_from) + " To: " + str(layer_to) + " condition: " + str_connect + " synapses: " + the_weights_str
                synapses[syn_no].connect(i=eval(the_from), j=eval(the_to))
                #print str(len(the_weights))
                for i in range (0, len(the_weights)):
                    #print eval(the_weights[i])
                    synapses[syn_no].w[i] = eval(the_weights[i]) 
                syn_no += 1

    # Declaring the monitors        
    stateMotor = StateMonitor(neurons[MOTOR_LAYER], 'v', record=True)
    spikeMonitor = SpikeMonitor(neurons[MOTOR_LAYER])
            
    # Integrtion of each component in the network. 
    if verbose: 
        rospy.loginfo("Integration of each component in the network.")
    net = Network(collect())
    net.add(neurons)
    net.add(synapses)
    
    if verbose:
        rospy.loginfo("Saving initialized SNN...")
    net.store(initFile, pathSNN+"learned/"+initFile+".dat")

    #if verbose:
    #    rospy.loginfo("Restoring previously learned SNN...")
    #net.restore(learnedFile, pathSNN+learnedFile+".dat")
    # Keep this initial state in memory.  Restore() after each simulation. 
    net.store("current")

    init_frames_in()
    
    # Main loop.  Inifite if RUN mode.   Quit after X iteration if LEARNING mode. 
    theExit = False
    while not theExit: 
        # Start the cycle and the timer.
        start = time.time()
        time.clock()
        if verbose:
            display_chrono(start, "BEGIN CYCLE")

        # Restore initial SNN and monitors
        net.restore("current")
        
        # When the callback function has received all the input neurons, assign those neurons to the input layer. 
        frames_assignation = frames_in
        #rospy.loginfo("Assigned sensories: " + str(frames_assignation))

        # Assing sensory neurons from frames              
        for k in range(0,sensory_neurons): 
            neurons[SENSORY_LAYER].v[k] = frames_assignation[k]   # Only v of the sensory neurons
            neurons[SENSORY_LAYER].I[k] = input_drive_current 
            rospy.loginfo("neuron " + str(k) + " v. " + str(neurons[SENSORY_LAYER].v[k])) 
        init_frames_in()

        # Simulation execution
        net.run(simulation_lenght_ms)
                      
        # Publish on the output topic
        for y in range(0, motor_neurons):
            rospy.loginfo("Values to publish for neuron " + str(y) + " : " + str(len(stateMotor.v[y])))
            rospy.loginfo("Number of spikes: " + str(spikeMonitor.num_spikes))
            # publish volts
            voltsToPublish = Float32MultiArray()
            voltsToPublish.data = stateMotor.v[y]
            topics_motor_volts[y].publish(voltsToPublish) 
            # publish spikes    
            nb_spikes = sum(spikeNo == y for spikeNo in spikeMonitor.i)
            topics_motor_spikes[y].publish(nb_spikes)
        topic_simulation_lenght.publish(simulation_lenght_int)
        #rospy.loginfo("Transmitted voltage values: "  + str(len(stateMotor.v[0])))

        # End of the cycle
        display_chrono(start, "END OF CYCLE")    
        rospy.loginfo("-----------")


if verbose:
    rospy.loginfo("Subscribe to the callbacks (input neurons)...")
for k in range(0, sensory_neurons):
    rospy.Subscriber("/"+SNNname+"_"+str(k+1)+"_snn_in", String, callbackReceiveMsgFromTopic, k)
    
for k in range(0, motor_neurons):
    topics_motor_volts.append(rospy.Publisher('motor_volts_'+SNNname+str(k+1), Float32MultiArray, queue_size=1))
    topics_motor_spikes.append(rospy.Publisher('motor_spikes_'+SNNname+str(k+1), Float32, queue_size=1))
topic_simulation_lenght = rospy.Publisher('simulation_lenght_'+SNNname, Int16, queue_size=1)
# Call the SNN system
SNN()

