function store_data(DataToSave){
$.ajax({
	type:'POST',
	url:'./php/processdata.php',
	data:{  Code: 					  DataToSave.code,
		    Name:  					  DataToSave.name,
 	 	    Trial:  				  DataToSave.trial,
 		    NumberOfItems:		 	  DataToSave.nstim,
	        WhichIsBigger:			  DataToSave.whichisbigger,
    	    CircleOrSquare:			  DataToSave.circleorsquare,
        	AreaPercValueSecondStim:  DataToSave.stimval,
        	AreaPercValueThirdStim:   DataToSave.stimval_3,
    	    PositionBig:			  DataToSave.posbig,
        	Area1:					  DataToSave.area1,
		    Area2:					  DataToSave.area2,
		    Area3:	 				  DataToSave.area3,
        	Angle:	 				  DataToSave.angle,
        	StepAngle:	 			  DataToSave.stepangle,
        	Response: 				  DataToSave.response,
        	Correct: 				  DataToSave.correct,
        	ReactionTime: 			  DataToSave.reactiontime,
		    Confidence: 			  DataToSave.confidence,
		    ReactionTimeConfidence:   DataToSave.reactiontimeconf},
  		    error: function (e)
  		    {alert("Server error - " + e);}
})
}
