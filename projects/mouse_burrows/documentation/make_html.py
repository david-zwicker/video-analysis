#!/usr/bin/env python2
'''
Created on Dec 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..' ))

from collections import OrderedDict

import grip

from projects.mouse_burrows.algorithm.parameters import PARAMETER_LIST, UNIT
from utils.data_structures import DictXpath


def create_parameter_summary(outfile):
    """ creates a markdown file summarizing the parameters """
    # load the parameters into a convenient structure
    parameters = DictXpath(dict_class=OrderedDict)
    deprecated = {}
    for parameter in PARAMETER_LIST:
        if parameter.unit is UNIT.DEPRECATED:
            deprecated[parameter.key] = parameter
        else:
            parameters[parameter.key] = parameter
            
    
    def param_line(parameter):        
        return "`%s = %s`: %s\n" % (parameter.key, parameter.default_value,
                                    parameter.description)     
    
    with open(outfile, 'w') as fp:
        # list all the important parameters
        fp.write('Parameters\n====\n')
        for key1, value1 in parameters.iteritems():
            if isinstance(value1, DictXpath):
                # we got a sub category
                category, subset = key1, value1
                fp.write("\n%s\n" % category)
                fp.write("%s\n" % ("-" * len(category)))

                for key2, value2 in subset.iteritems():
                    if isinstance(value2, DictXpath):
                        fp.write("* `%s`\n" % key2)
                        for parameter in value2.itervalues():
                            fp.write("  * " + param_line(parameter))
                    else:
                        fp.write("* " + param_line(value2))
            
            else:
                fp.write("* " + param_line(value1))
                
        # list the deprecated parameters
        fp.write('\nDeprecated parameters\n====\n')
        for _, parameter in sorted(deprecated.iteritems()):
            fp.write("* " + param_line(parameter))
    
    
create_parameter_summary('parameters.md')

grip.command.main(['parameters.md', '--export', 'html/parameters.html'])
grip.command.main(['statistics.md', '--export', 'html/statistics.html'])