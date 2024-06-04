import copy
import re

from lxml import etree
from lxml import html
salient_attributes = {
    "alt",
    "aria_description",
    "aria_label",
    "aria_role",
    "input_checked",
    "input_value",
    "label",
    "name",
    "option_selected",
    "placeholder",
    "role",
    "text_value",
    "title",
    "type",
    "value",
} # fixme: did not include the href and src

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_outerhtml(element):
    # Convert the element to string
    element_string = html.tostring(element).decode('utf-8')

    # Extract the outerHTML
    start_tag = element_string.split('>')[0] + '>'
    inner_html = ''.join(element_string.split('>')[1:-1])
    if isinstance(element.tag, str):
        outer_html = start_tag + inner_html + '</' + element.tag + '>'
        return outer_html
    else:
        return ''

def get_descendants(node, max_depth, current_depth=0):
    '''get all descendants'''
    if current_depth > max_depth:
        return []
    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))
    return descendants

def get_dom_path(element):
    tree = element.getroottree()
    path = tree.getpath(element)
    return path

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


def clean_tree(dom_tree, all_candidate_ids):
    '''clean the tree by removing unimportant elements and attributes'''

    new_tree = copy.deepcopy(dom_tree)

    for node in new_tree.xpath("//*")[::-1]: # from leaf to root
        # check if node have salient attributes
        for attr in node.attrib:

            if attr == "class" and node.attrib[attr] and node.tag == "svg":
                icon_texts = re.findall(r"\S*icon\S*", node.attrib[attr], re.IGNORECASE)
                icon_texts = [clean_text(text) for text in icon_texts]
                icon_texts = [text for text in icon_texts if text]
                if icon_texts:
                    node.attrib[attr] = " ".join(icon_texts)
                else:
                    node.attrib.pop(attr) # remote unimportant attribute

            elif attr in salient_attributes:
                if not (
                    (
                        attr == "role" and node.attrib.get(attr, "") in {"presentation", "none", "link"}
                    )
                    or (attr == "type" and node.attrib.get(attr, "") == "hidden")
                ):
                    value = clean_text(node.attrib[attr])
                    if value != "":
                        node.attrib[attr] = value
                    else:
                        node.attrib.pop(attr)
                else:
                    node.attrib.pop(attr)

            elif attr != "backend_node_id":
                node.attrib.pop(attr)

        if node.tag == "text":
            value = clean_text(node.text)
            if len(value) > 0:
                node.text = value
            else: # empty <text>, remote the node
                node.getparent().remove(node)

        elif (
            node.attrib.get("backend_node_id", "") not in all_candidate_ids
            and len(node.attrib) == 1
            and not any([x.tag == "text" for x in node.getchildren()])
            and node.getparent() is not None
            and len(node.getchildren()) <= 1
        ): # a sparse leaf
            # insert all children into parent, and remove the node
            for child in node.getchildren():
                node.addprevious(child)
            node.getparent().remove(node)

    return new_tree


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
    # fixme: get siblings within range
    # parent = candidate_node.getparent()
    # if parent is not None:
    #     siblings = [x for x in parent.getchildren() if x.tag != "text"]
    #     idx_in_sibling = siblings.index(candidate_node)
    #     for x in siblings[
    #              max(0, idx_in_sibling - max_sibling): min(idx_in_sibling + max_sibling + 1, len(siblings))
    #              ]:
    #         nodes_to_keep.append(
    #             (
    #                 get_dom_path(x),
    #                 get_outerhtml(x)
    #             )
    #         )

    return outerhtml+')'*ct




