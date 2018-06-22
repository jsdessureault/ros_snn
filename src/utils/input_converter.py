#!/usr/bin/env python

'''
Filename: input_converter.py
Author: Jean-Sebastien Dessureault
Date created: 19/05/2017
Python version 2.7
'''

import rospy
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import *
import string
import csv

# Registering node to ROS
node = "node_converter_input"
rospy.init_node(node, anonymous=True)
rospy.loginfo(node)

# Retriving parameters from launcher
verbose = rospy.get_param("~verbose") 
SNNname = rospy.get_param("~SNNname")
pathSNN = rospy.get_param("~path")
topics_to_convert = rospy.get_param("~topics_to_convert")
p_topic = []
p_type = []
p_field = []
p_min = []
p_max = []
p_stringfile = []
out_topic = []


for x in range (0, topics_to_convert):
    p_topic.append(rospy.get_param("~input_topic_"+str(x+1))) 
    p_type.append(rospy.get_param("~topic_type_"+str(x+1)))
    p_field.append(rospy.get_param("~input_field_"+str(x+1)))
    p_min.append(rospy.get_param("~input_min_"+str(x+1)))
    p_max.append(rospy.get_param("~input_max_"+str(x+1)))
    p_stringfile.append(rospy.get_param("~stringfile_"+str(x+1)))

def MinMax(value_to_normalize, min, max):
    normalized_value = float((value_to_normalize - min) / (max - min))
    return normalized_value

def callback(data, neuron_nb):
    #rospy.loginfo(data)
    field = eval("data"+p_field[neuron_nb])
    if (p_stringfile[neuron_nb] == ""):
        # Numeric field
        converted_value = float(field)
        converted_value = MinMax(converted_value, p_min[neuron_nb], p_max[neuron_nb])
        #rospy.loginfo("Le callback publie la valeur %f sur le topic.", converted_value)
        out_topic[neuron_nb].publish(str(converted_value))
        #print "Publie: " + str(converted_value) + " sur " + str(neuron_nb)
    else:
        # string field
        csv_file = pathSNN + "csv/" + p_stringfile[neuron_nb]
	#print csv_file
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=",", quotechar="|")
            try:
                for row in reader:
                    if (row['string'] == str(field)):
			#print "publish: " + str(row['voltage'])
                        out_topic[neuron_nb].publish(str(row['voltage']))
            except csv.Error as e:
                 rospy.logerror("A string received in the converter parameter (input_parameter.py) has no match in the specified csv file. Giving 0.0 value.")
                 out_topic[neuron_nb].publish("0.0")
    
if verbose:
    rospy.loginfo("Mapping and converting topics for sensory inputs of the SNN.  Do NOT interrupt while SNN execute. ")

for x in range (0, topics_to_convert):
    rospy.Subscriber(p_topic[x], eval(p_type[x]) , callback, x)
    out_topic.append(rospy.Publisher('/'+SNNname+'_'+str(x+1) + "_snn_in", String, queue_size=1))
    #print "Declaration du topic: " + '/ssn_in_'+SNNname+'_'+str(x+1)

rospy.spin()
