#include <stdio.h>       
#include <stdlib.h>
#include <string.h> /*for memcpy command*/
#include "movieobj.h"
#include "video.h"

#include <mem.h>
#include "exptlib.h"


#define status STAND_ALONE
#define numblocks 22
#define backup 1
#define fore_time 30
#define delta 11
#define chunk 1
#define wide 0

void setstim(int *stim, long *seed, int *trialsperblock,int *blocktypes,int numtrials);
void hline(int len, unsigned char far *b,long *seed);
void vhline(int len, int y, unsigned char far *b,long *seed);


int tasktypes[numblocks]={1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4};  
int blocktypestims[4][7]={{0,1},{2,3,4},{5,6},{0,1,2,3,4,5,6}};
int blockstimnums[4]={2,3,2,7};

int lengthwide[7]={23,39,99,123,148,230,260};
int lengthnarrow[7]={57,77,99,123,148,174,201};
int trialsperblockchunk[numblocks]={35,35,35,35,35,35,36,36,36,36,36,36,36,36,36,36,36,36,35,35,35,35};
int blocktypeschunk[numblocks]={4,4,4,4,4,4,1,2,3,1,2,3,1,2,3,1,2,3,4,4,4,4};
int numtrialschunk=782;
int trialsperblocknochunk[numblocks]={35,35,35,35,35,35,37,37,36,36,36,36,36,36,36,36,36,36,35,35,35,35};
int blocktypesnochunk[numblocks]={4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};
int numtrialsnochunk=784;



void main()
{                  
  int length[7];
  int trialsperblock[numblocks];
  int blocktypes[numblocks];
  int numtrials;
  int *stim;
  int b,s,t,i;
  int stimulus;
  logtype *log;
  FILE *outfile,*backfile;
  char *fontlist,junk[3];
  char backstr[40],outline[80];
  long seed;

  image *wrong_i,*blank, *targ_i;
  movie *wrong_m,*ready_m,*stim_m,*feed_m;
  image *ready7_i,*readya_i,*readyb_i,*readyc_i;
  response *data; int resp; long RT;

  fontlist=getenv("FONTLIST");
  log=(logtype *)malloc(1*sizeof(logtype));
  gen_init(status,log);
  SetupMoviePackage(fontlist);
  seed=log->seed;
  outfile=fopen(log->outfn,"wt");
  if (backup)
    {
     sprintf(backstr,"c:\\expbak\\%s",log->outfn);
     backfile=fopen(backstr,"wt");
    }

if (wide){
for (i=0;i<7;i++){
length[i]=lengthwide[i];
}
}else{
for (i=0;i<7;i++){
length[i]=lengthnarrow[i];
}
}

if (chunk){
 for (i=0;i<numblocks;i++){
 trialsperblock[i]=trialsperblockchunk[i];
 blocktypes[i]=blocktypeschunk[i];
 }
 numtrials=numtrialschunk;
}else{
 for (i=0;i<numblocks;i++){
  trialsperblock[i]=trialsperblocknochunk[i];
  blocktypes[i]=blocktypesnochunk[i];
  }
  numtrials=numtrialsnochunk;
}


  stim= (int *)malloc(sizeof(int)*numtrials);
  setstim(stim,&seed,trialsperblock,blocktypes,numtrials); 
  makePalette(GRAYSCALE);
  /*READY movie*/
  /*wrong movie*/
  wrong_i=newImage();
  downloadImage(wrong_i);
  drawText("invalid key",0,0,0,255);
  uploadImage(wrong_i);
  wrong_m=initMovie(1);
  setMovie(wrong_m,0,wrong_i,24);

  ready7_i=newImage();
  downloadImage(ready7_i);
  for (s=0;s<7;s++)
  {
    sprintf(junk,"%d",s+1);
    vhline(length[s],(s*16),picBuf,&seed);
    drawText(junk,xcenter,(s*16)+2,0,255);
  }
  uploadImage(ready7_i);

  readya_i=newImage();
  downloadImage(readya_i);
  for (s=0;s<2;s++)
  {
    sprintf(junk,"%d",s+1);
    vhline(length[s],(s*16),picBuf,&seed);
    drawText(junk,xcenter,(s*16)+2,0,255);
  }
  uploadImage(readya_i);
  
  readyb_i=newImage();
  downloadImage(readyb_i);
  for (s=2;s<5;s++)
  {
    sprintf(junk,"%d",s+1);
    vhline(length[s],(s*16),picBuf,&seed);
    drawText(junk,xcenter,(s*16)+2,0,255);
  }
  uploadImage(readyb_i);
  
  readyc_i=newImage();
  downloadImage(readyc_i);
  for (s=5;s<7;s++)
  {
    sprintf(junk,"%d",s+1);
    vhline(length[s],(s*16),picBuf,&seed);
    drawText(junk,xcenter,(s*16)+2,0,255);
  }
  uploadImage(readyc_i);


  ready_m=initMovie(1);



  stim_m=initMovie(2);
  feed_m=initMovie(1);
  blank=newImage();
  targ_i=newImage();
  setMovie(stim_m,0,blank,fore_time);

t=0;
  for (b=0;b<numblocks;b++)
    {
      switch (blocktypes[b]){
	  case 2: setMovie(ready_m,0,readyb_i,1);break;
	  case 3: setMovie(ready_m,0,readyc_i,1);break;
	  case 4: setMovie(ready_m,0,ready7_i,1);break;
	  default : setMovie(ready_m,0,readya_i,1);break;}
      runMovie(ready_m,UNTIL_RESPONSE,1);
      for (s=0;s<trialsperblock[b];s++)
	{
	  downloadImage(targ_i);
	  clearPicBuf();
	  stimulus=blocktypestims[blocktypes[b]-1][stim[t]];
	  hline(length[stimulus],picBuf,&seed);
	  uploadImage(targ_i);
	  setMovie(stim_m,1,targ_i,1);
	  data=runMovie(stim_m,UNTIL_RESPONSE,1);
	  RT=data->x[0].rt-(msperframe*fore_time);
	  switch (data->x[0].resp){
	  case '1': resp=0;break;
	  case '2': resp=1;break;
	  case '3': resp=2;break;
	  case '4': resp=3;break;
	  case '5': resp=4;break;
	  case '6': resp=5;break;
	  case '7': resp=6;break;
	  case '@': resp=7;break;
	  default : resp=8;audio(INVALID,wrong_m);break;}
	  if (resp==stimulus) audio(CORRECT);
	  else
	    {
	      sprintf(junk,"%d",stimulus+1);
	      downloadImage(targ_i);
	      drawText(junk,xcenter-10,ycenter+60,0,255);
	      uploadImage(targ_i);
	      setMovie(feed_m,0,targ_i,30);
	      runMovie(feed_m,FULL_SHOW,0);
	    }
	  sprintf(outline,"sub%03d ch%d wd%d blk%02d trl%03d bt%1d set%d %02d %02d %05d\n",
		  log->subjnum,chunk,wide,b,s,tasktypes[b],blocktypes[b],stimulus,resp,RT);
	  fprintf(outfile, "%s",outline);
	  if (backup) fprintf(backfile,"%s",outline);
	  if (resp==7) {fclose(outfile);CleanupMoviePackage();
	  printf ("stopped while running by participant\n");exit(1);}
	  t++;
	}
    }
  fclose(outfile);
  if (backup) fclose(backfile);
  thankyou();
  CleanupMoviePackage();
}








void vhline(int len, int y, unsigned char far *b,long *seed)
{
  int q, yrnd,xstart;
  char junk[3];
  
  yrnd=y;
  xstart=xcenter-(len/2);
  for (q=xstart;q<xstart+len;q++) b[o(q,yrnd)]=255;
}



void hline(int len, unsigned char far *b,long *seed)
{
  int q, yrnd,xstart;
  char junk[3];
  
  yrnd=ycenter+randint(-delta,delta,seed);
  xstart=xcenter-(len/2)+randint(-delta,delta,seed);
  for (q=xstart;q<xstart+len;q++) b[o(q,yrnd)]=255;
}


void setstim(int *stim, long *seed, int *trialsperblock, int *blocktypes, int numtrials)
{
  
  int i,j,k,c=0;
  int temp[434];

  for (i=0;i<numblocks;i++){
    for (j=0;j<trialsperblock[i];j++){
       temp[j]=j%(blockstimnums[blocktypes[i]-1]);
    }
    distribute(temp,trialsperblock[i],seed);
    for (j=0;j<trialsperblock[i];j++){
      stim[c++]=temp[j];
    }
  }
  if (!chunk){
	for (i=0;i<434;i++){
		temp[i]=i%7;
	}
	distribute(temp,434,seed);
	for (i=0;i<434;i++){
		stim[210+i]=temp[i];
	}
  }
  for (i=0;i<140;i++)
    if(chunk){
      stim[644+i]=stim[70+i];
    }else{
      stim[642+i]=stim[70+i];
    }
}
 
