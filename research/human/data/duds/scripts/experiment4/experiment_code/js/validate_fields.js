function validateFields()
{	var Name 		= document.getElementById("Name").value;
	var Gender 		= document.getElementById("Gender").value;
	var Age 		= document.getElementById("Age").value;
	var Checkbox 	= document.getElementById("consent_checkbox").checked;
	var Input_type 	= $('[name=Input]:checked').length;
	var Email       = $('#Contacto').val()

	var Emailtest = true;
	if(Email.trim() != ""){
		const re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
		Emailtest =  re.test(String(Email).toLowerCase());
	}

	if ( ((Age>=18) && (Age<99)) && ((Gender == "f") || (Gender == "m") || (Gender == "noBinario") || (Gender == "Pref_no_responder")) && (Name.length>3) && Checkbox && (Input_type) && (Emailtest))
	{
		$("#submit_button").fadeTo(.1, 1);
		$("#submit_button").prop('disabled', false);
	}
	else
	{
		$("#submit_button").fadeTo(.1, .5);
		$("#submit_button").prop('disabled', true);
	}
}

