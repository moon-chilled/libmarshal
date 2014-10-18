void Heuristic33(int* Aout, int* aout, int* Bout, int* bout, size_t* hf_sorted, size_t* wf_sorted, std::vector<int> hoptions, std::vector<int> woptions, int h, int w){
  // Our heuristic
  int A = 0; int a = 0; int B = 0; int b = 0;
  struct int2{int x; int y;};
  int k = 0; int l = 0; int p = 0;
  int2 maxtile; maxtile.x = 0; maxtile.y = 0;
  int re = 0; int done = 0;
#if SP
  int min_limit = 24;
  int max_limit = 3040;
#else
  int min_limit = 48;
  int max_limit = 1520;
#endif
  int hoptions_good[hoptions.size()];
  int woptions_good[woptions.size()];
  int2 tileoptions[hoptions.size()*woptions.size()];
do{
  k = 0; l = 0; p = 0; re = 0;
  // Desired minimum and maximum for a and b
  for (int j = 0; j < hoptions.size(); j++)
    if (hoptions[hf_sorted[j]] >= min_limit && hoptions[hf_sorted[j]] <= max_limit){
      hoptions_good[k] = hoptions[hf_sorted[j]];
      k++;
    }
  for (int jj = 0; jj < woptions.size(); jj++){
    //if (woptions[wf_sorted[jj]] >= min_limit && woptions[wf_sorted[jj]] <= max_limit){
      woptions_good[l] = woptions[wf_sorted[jj]];
      l++;
  }
  // Two in the desired range
  if (k > 0 && l > 0){
    for (int i = 0; i < k; i++)
        if (hoptions_good[i] * w < MAX_MEM){ // Fits in local memory
          tileoptions[p].x = hoptions_good[i];
          for (int j = 0; j < l; j++)
#if SP
	        if (tileoptions[p].x * woptions_good[j] < (MAX_MEM-64)/2) // Fits in local memory
#else
	        if (tileoptions[p].x * woptions_good[j] < (MAX_MEM-32)/2) // Fits in local memory
#endif
                  tileoptions[p].y = woptions_good[j];
                else
                  tileoptions[p].y = 1;
          p++;
        }

    int maxfactor = 1;
    for (int j = 0; j < p; j++){ // Use as much local memory as possible
      done = 1;
      int tilesize = tileoptions[j].x * tileoptions[j].y;
      int factor = 1;
      if (tilesize < MAX_MEM/16) factor = 16;
      else if (tilesize >= MAX_MEM/16 && tilesize < MAX_MEM/8) factor = 8;
      else if (tilesize >= MAX_MEM/8 && tilesize < MAX_MEM/4) factor = 4;
      else if (tilesize >= MAX_MEM/4 && tilesize < MAX_MEM/2) factor = 2;
      tilesize *= factor;
      if (tilesize > maxtile.x * maxtile.y * maxfactor){
        maxtile.x = tileoptions[j].x;
        maxtile.y = tileoptions[j].y;
        maxfactor = factor;
      }
    }
    if (p == 0 && min_limit <= 0) // Does not fit in local memory: largest a and b possible
      for (int i = 0; i < k; i++)
          if (((hoptions_good[i]*w+31)/32) + ((((hoptions_good[i]*w+31)/32)>>5)*1) <= MAX_MEM){
            maxtile.x = hoptions_good[i];
            for (int j = re; j < l; j++){
#if SP
              if (maxtile.x * woptions_good[j] < (MAX_MEM-64)/2) // Fits in local memory
#else
              if (maxtile.x * woptions_good[j] < (MAX_MEM-32)/2) // Fits in local memory
#endif
                maxtile.y = woptions_good[j];
              else
                maxtile.y = 1;
              re = j;
              done = 1;
            }
          }
    if (p == 0){
      min_limit -= 2;
#if SP
      max_limit += 128; //256
#else
      max_limit += 256;
#endif
    }
    if (done == 1){
      a = maxtile.x; A = h/a;
      b = maxtile.y; B = w/b;
    }
  }
  else{
    min_limit -= 2;
    max_limit += 256;
  }
}while(!done && min_limit >= 0);
  k = 0; l = 0;
  // None in the desired range
  if (done == 0){
    for (int j = 0; j < hoptions.size(); j++)
#if SP
      if (hoptions[hf_sorted[j]] <= 6112)
#else
      if (hoptions[hf_sorted[j]] <= 3056)
#endif
        k++;
    for (int j = 0; j < woptions.size(); j++)
#if SP
      if (woptions[wf_sorted[j]] <= 3000)
#else
      if (woptions[wf_sorted[j]] <= 1500)
#endif
        l++;
    if (k > 0){
      a = hoptions[hf_sorted[k-1]]; A = h/a;
    }
    else{
      a = 1; A = h/a;
    }
    if (l > 0){
      b = woptions[wf_sorted[l-1]]; B = w/b;
    }
    else{
      b = 1; B = w/b;
    }
  }

  *Aout = A; *aout = a; *Bout = B; *bout = b;
}

