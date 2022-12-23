function attemptRegister() {
    //Getting elements
    var email = document.getElementById("email").value;

    //Sending Post request to servers internal API
    if (email == "") {return;}
    var request = new XMLHttpRequest();
    request.open("POST", "/api/internal/register", true);
    request.setRequestHeader('Content-Type', 'application/json');
    request.send(JSON.stringify({
        "email": email,
    }));
    request.onload = function() {
        var data = JSON.parse(this.responseText);
        if (data["success"] != true) {
            showAlert(data["message"], "red");
        } else {
            showAlert("Your account is now registered..", "#00aced");
            //window.location.reload(false);
        }
    };
};

// Show an alert at the bottom signifying results of action
function showAlert(message, color) {
  var x = document.getElementById("sub");
  x.innerHTML = message;
  x.className = "show";
  x.style.backgroundColor  = color;
  setTimeout(function(){ x.className = x.className.replace("show", ""); }, 3000);
};

// Attempts to search for a value within page and highlight it whilst jumping to it
function buttonCode() {
    str = document.getElementById('search').value;
    var strFound;
    if (window.find) {
        var original_content = window;
        strFound=original_content.find(str);
        if (!strFound) {
            strFound=original_content.find(str,0,1);
            while (original_content.find(str,0,1)) continue;
        }
    }
    if (!strFound) showAlert("String '"+str+"' not found!", "red");
    return;
};