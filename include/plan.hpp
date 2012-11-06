class Transposition {
 public:
  const int m, n; // M by N in row major. i.e. A[i,j]=i*N+j and i<M
  Transposition(int mm, int nn): m(mm), n(nn), mn_(mm*nn-1),
  nr_cycles_(-1) {}
  unsigned Next(unsigned i) const {
    return (i*m)-mn_* (i/n);
  }
  unsigned GetNumCycles() {
    if (nr_cycles_ >= 0)
      return nr_cycles_;
    unsigned nontrivial = 0;
    return nontrivial;
  }
 private:
  int nr_cycles_;
  const int mn_;
  Transposition();
};

