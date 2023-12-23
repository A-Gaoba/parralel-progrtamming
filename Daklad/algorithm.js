  #pragma omp parallel
  { 
  for(int ic=0; ic<NC/*Number of colors*/; ++ic){ // цикл по цветам
  #pragma omp for
  for(int ie=C[ic]; ie<C[ic+1]; ++ie){ // цикл по ребрам графа данного цвета
  double r = Calc(ie);
  X[E[ie].v[0]] += r;
  X[E[ie].v[1]] += r; 
  }
  #pragma omp barrier // для наглядности (в omp for неявный барьер уже есть)
  }
  }