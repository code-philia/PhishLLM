////// Kudos to SSOScan for the scripts! /////
//      _onTopLayer_func_script             //
//      _isChildElement_func_script        //
////////////////////////////////////////////

// Override alerts
window.alert = null;

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


function get_dompath(e){
    if(e.parentNode==null || e.tagName=='HTML') return'';
    if(e===document.body || e===document.head) return'/'+e.tagName;
    for (var t=0, a=e.parentNode.childNodes, n=0; n<a.length; n++){
        var r=a[n];
        if(r===e) return get_dompath(e.parentNode)+'/'+e.tagName+'['+(t+1)+']';
        1===r.nodeType&&r.tagName===e.tagName&&t++}
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


var get_element_full_src=function(el){
    return el.outerHTML;
}


var get_all_links = function(){
    var links = document.getElementsByTagName('a');
    returned_links = [];
    for (let link of links){
        source = link.getAttribute("href");
        if (source != null) {
            returned_links.push([link, '//html' + get_dompath(link), source]);
        }
        else{
            returned_links.push([link, '//html' + get_dompath(link), "#"]);
        }
    }
    return returned_links;
}

var get_all_buttons=function(){
    var buttons = document.getElementsByTagName("button");
    returned_buttons = [];
    for (let button of buttons){
        returned_buttons.push([button, '//html'+get_dompath(button)]);
    }

    var all_elements = document.getElementsByTagName("input")
    for (let element of all_elements) {
        var [nodetag, etype, el_src, aria_label, eplaceholder, evalue, onclick, id, name, action] = get_element_properties(element);
        if (nodetag == "button" || etype == "submit" || etype == "button" || etype == "image" || onclick || etype == "reset") {
            returned_buttons.push([element, '//html' + get_dompath(element)]);
        } else if ('getAttribute' in element && element.getAttribute("data-toggle") === "modal") {
            returned_buttons.push([element, '//html' + get_dompath(element)]);
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
            returned_imgs.push([img, '//html'+get_dompath(img)]);
        }
    }
    return returned_imgs;
}

