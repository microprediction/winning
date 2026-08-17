function trials_structure(){

  TrialStruct = []

  var NSTIM               = new Array();
  var STIMVALUES          = new Array();
  var WHICHISBIGGER       = new Array();
  var TOMULTIPLY          = new Array();

  cont = 0;
  for (var a = 0; a < numberofstimuli.length; a++){
    for (var b = 0; b < areavalues.length; b++){
      for (var c = 0; c < bigger.length; c++){
        for (var d = 0; d < areaforthethirdone.length; d++){

          NSTIM[cont]          = numberofstimuli[a]
          STIMVALUES[cont]     = areavalues[b]
          WHICHISBIGGER[cont]  = bigger[c]
          TOMULTIPLY[cont]     = areaforthethirdone[d]
          cont = cont + 1

        }
      }
    }
  }

  sc.maxtrials = NSTIM.length;  //NSTIM.length;  volver a poner esto para que tire los 120 trials
  
  var indxs = new Array();
  for (var i = 0; i < NSTIM.length;i++) {indxs.push(i);}

  shuffle(indxs)
  for (var i = 0; i < sc.maxtrials; i++){
    sc.nstim[i]      			= NSTIM[indxs[i]]
    sc.stimvalues[i]      = STIMVALUES[indxs[i]]
    sc.whichisbigger[i]  	= WHICHISBIGGER[indxs[i]]
    sc.tomultiply[i]  		= TOMULTIPLY[indxs[i]]}

}
