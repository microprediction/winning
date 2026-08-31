#include <stdio.h>       
#include <stdlib.h>
#include <string.h> /*for memcpy command*/
#include "movieobj.h"
#include "video.h"

#include <mem.h>
#include "exptlib.h"


#define status STAND_ALONE
#define numtrials 50
#define trialsperblock 50
#define numblocks ((numtrials)/(trialsperblock))
#define backup 0
#define fore_time 30
#define delta 5

void hline(int len, unsigned char far *b,long *seed);


int length[2];

main()
{
  int *stim,*blktype;
  int b,s,t,correct=0;


  logtype *log;
  FILE *outfile,*backfile;
  char *fontlist,junk[3];
  char backstr[40],outline[80];
  long seed;

  image *wrong_i,*ready_i,*blank, *targ_i;
  movie *wrong_m,*ready_m,*stim_m,*feed_m;
  response *data; int resp; long RT;

  printf("Enter base length 1\n");
  scanf("%d",&(length[0]));
  gets(NULL);
  printf("Enter base length 2\n");
  scanf("%d",&(length[1]));
  gets(NULL);
  printf("%d %d \n",length[0],length[1]);
  

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

  stim= (int *)malloc(sizeof(int)*numtrials);
  for (t=0;t<numtrials;t++)
    stim[t]=t%2;
  distribute(stim,numtrials,&seed);

 /* set up pallete here*/
  makePalette(GRAYSCALE);
  /*READY movie*/
  ready_i=newImage();
  downloadImage(ready_i);
  drawText("ready?",0,0,0,255);
  uploadImage(ready_i);
  ready_m=initMovie(1);
  setMovie(ready_m,0,ready_i,1);
  /*wrong movie*/
  wrong_i=newImage();
  downloadImage(wrong_i);
  drawText("invalid key",0,0,0,255);
  uploadImage(wrong_i);
  wrong_m=initMovie(1);
  setMovie(wrong_m,0,wrong_i,24);

  stim_m=initMovie(2);
  feed_m=initMovie(1);
  blank=newImage();
  targ_i=newImage();
  setMovie(stim_m,0,blank,fore_time);



  for (b=0;b<numblocks;b++)
    {
      runMovie(ready_m,UNTIL_RESPONSE,1);
      for (s=0;s<trialsperblock;s++)
	{
	  t=s+(b*(trialsperblock));
	  downloadImage(targ_i);
	  clearPicBuf();
	  hline(length[stim[t]],picBuf,&seed);
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
	  case '8': resp=7;break;
	  case '9': resp=8;break;
	  case '0': resp=9;break;
	  case '-': resp=10;break;
	  case '=': resp=11;break;
	  case '@': resp=12;break;
	  default : resp=13;audio(INVALID,wrong_m);break;}
	  if (resp==stim[t]) 
	    {
	      audio(CORRECT);
	      correct++;
	    }
	  else
	    {
	      sprintf(junk,"%d",(stim[t])+1);
	      downloadImage(targ_i);
	      drawText(junk,xcenter-10,ycenter+60,0,255);
	      uploadImage(targ_i);
	      setMovie(feed_m,0,targ_i,30);
	      runMovie(feed_m,FULL_SHOW,0);
	    }
	  sprintf(outline,"sub%03d  blk%02d  trl%03d  %02d %02d %05d\n",
		  log->subjnum,b,s,stim[t],resp,RT);
	  fprintf(outfile, "%s",outline);
	  if (backup) fprintf(backfile,"%s",outline);
	  if (resp==12) {fclose(outfile);CleanupMoviePackage();
	  printf ("stopped while running by participant\n");exit(1);}
	}
    }
      fclose(outfile);
	if (backup) fclose(backfile);
  thankyou();
  CleanupMoviePackage();
  printf("%d \n",correct);
}


	  





  /*-----------------------------------------*/













void hline(int len, unsigned char far *b,long *seed)
{
  int q, yrnd,xstart;
  char junk[3];
  
  yrnd=ycenter+randint(-delta,delta,seed);
  xstart=xcenter-(len/2)+randint(-delta,delta,seed);
  for (q=xstart;q<xstart+len;q++) b[o(q,yrnd)]=255;
}


