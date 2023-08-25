import copy
import re
from lxml import etree
from lxml import html

def get_descendants(node, max_depth, current_depth=0):
    '''get all descendants'''
    if current_depth > max_depth:
        return []
    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))
    return descendants


def get_attribute_repr(node, max_value_length=5, max_length=20):
    # get attribute values in order
    attr_values_set = set()
    attr_values = ""
    for attr in [
        "class",
        "type",
        "href",
        "src",
        "value",
        "role",
        "aria_role",
        "aria_label",
        "label",
        "title",
        "aria_description",
        "name",
        "text_value",
        "input_checked",
    ]:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            # less menaingful values
            if value in [
                "hidden",
                "none",
                "presentation",
                "null",
                "undefined",
            ]:
                continue
            value = value.split() # split by space?
            # value = " ".join([v for v in value if len(v) < 15][:max_value_length])
            value = " ".join([v for v in value][:max_value_length])
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    return attr_values



def prune_tree(
    dom_tree,
    dom_path,
    max_depth=5,
    max_children=50,
    max_sibling=3,
):
    ''' get the context DOM tree for interested nodes'''
    outerhtml = ''
    try:
        candidate_node = dom_tree.xpath(dom_path.lower())[0]
    except (IndexError, etree.XPathEvalError):
        return
    outerhtml += '(' + candidate_node.tag  + ' ' + get_attribute_repr(candidate_node) # the dom path itself, content

    ct = 1
    # get descendants with max depth
    for x in get_descendants(candidate_node, max_depth)[:max_children]:
        if not isinstance(x.tag, str):
            continue
        outerhtml += '(' + x.tag + ' ' + get_attribute_repr(x)
        ct += 1


    return outerhtml+')'*ct




