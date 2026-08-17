function trials_structure(){

  TrialStruct = []

  var NSTIM               = new Array();
  var DIST12_VALUES       = new Array();
  
  cont = 0;
  for (var a = 0; a < numberofstimuli.length; a++){
    for (var b = 0; b < Distance_ratio_1_2.length; b++){
      for (var c = 0; c < NREP; c++){

          NSTIM[cont]          = numberofstimuli[a]
          DIST12_VALUES[cont]  = Distance_ratio_1_2[b]
          cont = cont + 1
      }
    }
  }
  
  sc.maxtrials = NSTIM.length;  // NSTIM.length tira todos los trials
//console.log(cont)
  
  var indxs = new Array();
  for (var i = 0; i < NSTIM.length;i++) {indxs.push(i);}

  shuffle(indxs)
  for (var i = 0; i < sc.maxtrials; i++){
    sc.nstim[i]      			= NSTIM[indxs[i]]
    
    sc.dist_12_values[i]  = DIST12_VALUES[indxs[i]]
  }
  }
