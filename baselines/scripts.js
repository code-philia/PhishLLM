////// Kudos to SSOScan for the scripts! /////
//      _onTopLayer_func_script             //
//      _isChildElement_func_script        //
////////////////////////////////////////////

// Override alerts
window.alert = null;

function getWindowSize(){
    return [window.innerWidth, window.innerHeight];
}

function get_loc(el){
    if ('getBoundingClientRect' in el) {
        var position = el.getBoundingClientRect();
        x1 = window.scrollX + position.left; // X
        y1 = window.scrollY + position.top; // Y
        x2 = window.scrollX + position.right; // X
        y2 = window.scrollY + position.bottom;// Y
        return [x1, y1, x2, y2];
    }
    else return [window.scrollX, window.scrollY, window.scrollX, window.scrollY];
}

function isNode(node) {
  return node && 'getAttribute' in node;
}

function onTopLayer(ele){ //check visibility
    if (!ele) return false;
	var document = ele.ownerDocument;
	var inputWidth = ele.offsetWidth;
	var inputHeight = ele.offsetHeight;
	if (inputWidth <= 0 || inputHeight <= 0) return false;
    if ('getClientRects' in ele && ele.getClientRects.length > 0) {
        var position = ele.getClientRects()[0];
        // console.log(position)
        var score = 0;
        position.top = position.top - window.pageYOffset;
        position.left = position.left - window.pageXOffset;
        var maxHeight = (document.documentElement.clientHeight - position.top > inputHeight) ? inputHeight : document.documentElement.clientHeight - position.top;
        var maxWidth = (document.documentElement.clientWidth > inputWidth) ? inputWidth : document.documentElement.clientWidth - position.left;
        for (j = 0; j < 10; j++) {
            score = isChildElement(ele, document.elementFromPoint(position.left + 1 + j * maxWidth / 10, position.top + 1 + j * maxHeight / 10)) ? score + 1 : score;
        }
        if (score >= 5) return true;
    }
    else return false;
}

function isChildElement(parent, child){
	if (child == null) return false;
	if (parent == child) return true;
	if (parent == null || typeof parent == 'undefined') return false;
	if (parent.children.length == 0) return false;
	for (i = 0; i < parent.children.length; i++){
        if (isChildElement(parent.children[i],child)) return true;
    }
	return false;
}

function getElementsByXPath(xpath, parent){
    let results = [];
    let query = document.evaluate(xpath,parent || document,null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
    for (let i=0, length=query.snapshotLength; i<length; ++i) {
        results.push(query.snapshotItem(i));
    }
    return results;
}

function getDescendants(node, accum) {
    var i;
    accum = accum || [];
    for (i = 0; i < node.childNodes.length; i++) {
        accum.push(node.childNodes[i]);
        getDescendants(node.childNodes[i], accum);
    }
    return accum;
}

function getAncestors(node) {
    if (node == null){
        return null;
    }
    if(node != document) {
        return [node].concat(getAncestors(node.parentNode)); //recursively get all parents upto document
    }
    else return [node];
}

function get_indexOf(array_elements, element){
    for(i = 0; i < array_elements.length; i++) {
        if(array_elements[i] == element){
            return i;
        }
    }
    return -1;
}

function findFirstCommonAncestor(nodeA, nodeB, ancestorsB) {
    var ancestorsB = ancestorsB || getAncestors(nodeB);
    if(nodeA == document){ // nodeA is the document already
        return nodeA;
    }
    else if(nodeB == document || ancestorsB.length == 0){ // nodeB is the document already or there is no ancestor for nodeB
        return nodeB;
    }
    else if(get_indexOf(ancestorsB, nodeA) > -1){ // get nodeA's index in the ancestorsB array
        return nodeA;
    }
    else return findFirstCommonAncestor(nodeA.parentNode, nodeB, ancestorsB);
}

function get_domdist(nodeA, nodeB){
    var common_ancestor = findFirstCommonAncestor(nodeA, nodeB);
    var ancestorsA = getAncestors(nodeA);
    var ancestorsB = getAncestors(nodeB);

    dist_to_A = get_indexOf(ancestorsA, common_ancestor);
    dist_to_B = get_indexOf(ancestorsB, common_ancestor);
    return [dist_to_A, dist_to_B];
}


function get_dompath(e){
    if(e.parentNode==null || e.tagName=='HTML') return'';
    if(e===document.body || e===document.head) return'/'+e.tagName;
    for (var t=0, a=e.parentNode.childNodes, n=0; n<a.length; n++){
        var r=a[n];
        if(r===e) return get_dompath(e.parentNode)+'/'+e.tagName+'['+(t+1)+']';
        1===r.nodeType&&r.tagName===e.tagName&&t++}
}

function get_dompath_nested(document_this, e){
    if(e.parentNode==null || e.tagName=='HTML') return'';
    if(e===document_this.body || e===document_this.head) return'/'+e.tagName;
    for (var t=0, a=e.parentNode.childNodes, n=0; n<a.length; n++){
        var r=a[n];
        if(r===e) return get_dompath_nested(document_this, e.parentNode)+'/'+e.tagName+'['+(t+1)+']';
        1===r.nodeType&&r.tagName===e.tagName&&t++}
}

function get_dom_depth_forelement(e, depth=1){
    if(e != document) {
        return depth + get_dom_depth_forelement(e.parentNode, depth); //recursively get all parents upto document
    }
    else return depth;
}

function get_dom_depth(){
    const getDepth = (node => {
      if (!node.childNodes || node.childNodes.length === 0) {
        return 1; //no child node
    }
    const maxChildrenDepth = [...node.childNodes].map(v => getDepth(v)); // get maximum childnodes depth
    return 1 + Math.max(...maxChildrenDepth);
    })
    return getDepth(document.documentElement); // dom depth for the document
}



function get_element_properties(element){
    var nodetag = element.tagName.toLowerCase();
    var etype = element.type;
    var el_src = get_element_full_src(element);

    var aria_label = null;
    var eplaceholder = null;
    var evalue = null;
    var onclick = null;
    var id = null;
    var name = null;
    var action = null;
    if ('getAttribute' in element) {
        aria_label = element.getAttribute("aria-label");
        eplaceholder = element.getAttribute("placeholder")
        evalue = element.getAttribute("value");
        onclick = element.getAttribute("onclick");
        id = element.getAttribute("id");
        name = element.getAttribute("name");
        action = element.getAttribute("action");
    }
    return [nodetag, etype, el_src, aria_label, eplaceholder, evalue, onclick, id, name, action];
}


function get_attributes(el){
    var items = {};
    for (index = 0; index < arguments[0].attributes.length; ++index){
        items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value
    } return items;
}


function isHidden(el) {
 var style = window.getComputedStyle(el);
 return ((style.display === 'none') || (style.visibility === 'hidden'))
}

var get_element_src=function(el){
    return el.outerHTML.split(">")[0]+">";
}

var get_element_full_src=function(el){
    return el.outerHTML;
}

var get_element_full_text=function(el){
    return el.innerText;
}



var get_all_links = function(){
    var links = document.getElementsByTagName('a');
    returned_links = [];
    for (let link of links){
        source = link.getAttribute("href");
        if (source != null) {
            returned_links.push('//html' + get_dompath(link));
        }
    }
    return returned_links;
}

var get_all_buttons=function(){
    var buttons = document.getElementsByTagName("button");
    returned_buttons = [];
    for (let button of buttons){
        returned_buttons.push('//html'+get_dompath(button));
    }

    var all_elements = document.getElementsByTagName("input")
    for (let element of all_elements) {
        var [nodetag, etype, el_src, aria_label, eplaceholder, evalue, onclick, id, name, action] = get_element_properties(element);
        if (nodetag == "button" || etype == "submit" || etype == "button" || etype == "image" || onclick || etype == "reset") {
            returned_buttons.push('//html' + get_dompath(element));
        } else if ('getAttribute' in element && element.getAttribute("data-toggle") === "modal") {
            returned_buttons.push('//html' + get_dompath(element));
        }
    }
    return returned_buttons;
}


var get_all_clickable_imgs = function(){
    var imgs = document.getElementsByTagName("img");
    returned_imgs = [];
    for (let img of imgs){
        if (!onTopLayer(img)) {
            continue; // invisible
        }
        var [nodetag, etype, el_src, aria_label, eplaceholder, evalue, onclick, id, name, action] = get_element_properties(img);
        if (onclick || (el_src && len(el_src)>0)) {
            returned_imgs.push('//html'+get_dompath(img));
        }
    }
    return returned_imgs;
}


console.log("XDriver lib is setup!");