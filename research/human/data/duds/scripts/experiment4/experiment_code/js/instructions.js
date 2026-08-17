    function instructions(nro)
{	for (var i=1;i<=4;i++) {document.getElementById("exp"+i).style.display="none";}
	if (nro!=0) {document.getElementById("exp"+nro).style.display="block";}
	window.scrollTo(0,0);
}

