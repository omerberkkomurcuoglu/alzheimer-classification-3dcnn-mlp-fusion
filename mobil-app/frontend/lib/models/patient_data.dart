/// Patient data model matching backend ClinicalData schema.
class PatientData {
  // Demographic
  double ageAtVisit;
  double animals;
  double asscmem;
  double srtfree;
  double srttotal;
  double tma;
  double tmb;

  // Living
  double independ;
  double residenc;
  double maristat;
  double insex;
  double inrelto;
  double inlivwth;
  double inrely;
  double anymeds;

  // Physical
  double weight;
  double bpsys;
  double bpdias;

  // NPI-Q
  double npiqinf;
  double del;
  double hall;
  double agit;
  double depd;
  double anx;
  double elat;
  double apa;
  double disn;
  double irr;
  double mot;
  double nite;
  double app;

  // GDS
  double satis;
  double dropact;
  double empty;
  double bored;
  double spirits;
  double afraid;
  double happy;
  double helpless;
  double stayhome;
  double wondrful;
  double wrthless;
  double energy;
  double hopeless;
  double better;
  double gds;

  // FAQ
  double bills;
  double taxes;
  double shopping;
  double games;
  double stove;
  double mealprep;
  double events;
  double payattn;
  double remdates;
  double travel;

  // Medical
  double psp;
  double cort;
  double hunt;
  double prion;
  double meds;
  double dysill;
  double dep;
  double othpsy;
  double downs;
  double park;
  double hyceph;
  double brninj;
  double neop;
  double cogoth;
  double cogoth2;
  double cogoth3;

  // Medication embeddings (32 values)
  List<double> medsEmbeddings;

  PatientData({
    this.ageAtVisit = 65,
    this.animals = 20,
    this.asscmem = 14,
    this.srtfree = 30,
    this.srttotal = 48,
    this.tma = 25,
    this.tmb = 60,
    this.independ = 1,
    this.residenc = 1,
    this.maristat = 1,
    this.insex = 1,
    this.inrelto = 1,
    this.inlivwth = 0,
    this.inrely = 0,
    this.anymeds = 1,
    this.weight = 170,
    this.bpsys = 120,
    this.bpdias = 80,
    this.npiqinf = 1,
    this.del = 0,
    this.hall = 0,
    this.agit = 0,
    this.depd = 0,
    this.anx = 0,
    this.elat = 0,
    this.apa = 0,
    this.disn = 0,
    this.irr = 0,
    this.mot = 0,
    this.nite = 0,
    this.app = 0,
    this.satis = 0,
    this.dropact = 0,
    this.empty = 0,
    this.bored = 0,
    this.spirits = 0,
    this.afraid = 0,
    this.happy = 0,
    this.helpless = 0,
    this.stayhome = 0,
    this.wondrful = 0,
    this.wrthless = 0,
    this.energy = 0,
    this.hopeless = 0,
    this.better = 0,
    this.gds = 0,
    this.bills = 0,
    this.taxes = 0,
    this.shopping = 0,
    this.games = 0,
    this.stove = 0,
    this.mealprep = 0,
    this.events = 0,
    this.payattn = 0,
    this.remdates = 0,
    this.travel = 0,
    this.psp = 0,
    this.cort = 0,
    this.hunt = 0,
    this.prion = 0,
    this.meds = 0,
    this.dysill = 0,
    this.dep = 0,
    this.othpsy = 0,
    this.downs = 0,
    this.park = 0,
    this.hyceph = 0,
    this.brninj = 0,
    this.neop = 0,
    this.cogoth = 0,
    this.cogoth2 = 0,
    this.cogoth3 = 0,
    List<double>? medsEmbeddings,
  }) : medsEmbeddings = medsEmbeddings ?? List.filled(32, 0.0);

  Map<String, dynamic> toJson() {
    return {
      'age at visit': ageAtVisit,
      'ANIMALS': animals,
      'asscmem': asscmem,
      'srtfree': srtfree,
      'srttotal': srttotal,
      'tma': tma,
      'tmb': tmb,
      'INDEPEND': independ,
      'RESIDENC': residenc,
      'MARISTAT': maristat,
      'INSEX': insex,
      'INRELTO': inrelto,
      'INLIVWTH': inlivwth,
      'INRELY': inrely,
      'anymeds': anymeds,
      'WEIGHT': weight,
      'BPSYS': bpsys,
      'BPDIAS': bpdias,
      'NPIQINF': npiqinf,
      'DEL': del,
      'HALL': hall,
      'AGIT': agit,
      'DEPD': depd,
      'ANX': anx,
      'ELAT': elat,
      'APA': apa,
      'DISN': disn,
      'IRR': irr,
      'MOT': mot,
      'NITE': nite,
      'APP': app,
      'SATIS': satis,
      'DROPACT': dropact,
      'EMPTY': empty,
      'BORED': bored,
      'SPIRITS': spirits,
      'AFRAID': afraid,
      'HAPPY': happy,
      'HELPLESS': helpless,
      'STAYHOME': stayhome,
      'WONDRFUL': wondrful,
      'WRTHLESS': wrthless,
      'ENERGY': energy,
      'HOPELESS': hopeless,
      'BETTER': better,
      'GDS': gds,
      'BILLS': bills,
      'TAXES': taxes,
      'SHOPPING': shopping,
      'GAMES': games,
      'STOVE': stove,
      'MEALPREP': mealprep,
      'EVENTS': events,
      'PAYATTN': payattn,
      'REMDATES': remdates,
      'TRAVEL': travel,
      'PSP': psp,
      'CORT': cort,
      'HUNT': hunt,
      'PRION': prion,
      'MEDS': meds,
      'DYSILL': dysill,
      'DEP': dep,
      'OTHPSY': othpsy,
      'DOWNS': downs,
      'PARK': park,
      'HYCEPH': hyceph,
      'BRNINJ': brninj,
      'NEOP': neop,
      'COGOTH': cogoth,
      'COGOTH2': cogoth2,
      'COGOTH3': cogoth3,
      for (int i = 0; i < 8; i++) 'meds_1_emb_$i': medsEmbeddings[i],
      for (int i = 0; i < 8; i++) 'meds_2_emb_$i': medsEmbeddings[8 + i],
      for (int i = 0; i < 8; i++) 'meds_3_emb_$i': medsEmbeddings[16 + i],
      for (int i = 0; i < 8; i++) 'meds_4_emb_$i': medsEmbeddings[24 + i],
    };
  }

  factory PatientData.fromJson(Map<String, dynamic> json) {
    List<double> embs = List.filled(32, 0.0);
    for (int m = 1; m <= 4; m++) {
      for (int d = 0; d < 8; d++) {
        final key = 'meds_${m}_emb_$d';
        if (json.containsKey(key)) {
          embs[(m - 1) * 8 + d] = (json[key] as num).toDouble();
        }
      }
    }

    return PatientData(
      ageAtVisit: (json['age at visit'] as num?)?.toDouble() ?? 65,
      animals: (json['ANIMALS'] as num?)?.toDouble() ?? 20,
      asscmem: (json['asscmem'] as num?)?.toDouble() ?? 14,
      srtfree: (json['srtfree'] as num?)?.toDouble() ?? 30,
      srttotal: (json['srttotal'] as num?)?.toDouble() ?? 48,
      tma: (json['tma'] as num?)?.toDouble() ?? 25,
      tmb: (json['tmb'] as num?)?.toDouble() ?? 60,
      independ: (json['INDEPEND'] as num?)?.toDouble() ?? 1,
      residenc: (json['RESIDENC'] as num?)?.toDouble() ?? 1,
      maristat: (json['MARISTAT'] as num?)?.toDouble() ?? 1,
      insex: (json['INSEX'] as num?)?.toDouble() ?? 1,
      inrelto: (json['INRELTO'] as num?)?.toDouble() ?? 1,
      inlivwth: (json['INLIVWTH'] as num?)?.toDouble() ?? 0,
      inrely: (json['INRELY'] as num?)?.toDouble() ?? 0,
      anymeds: (json['anymeds'] as num?)?.toDouble() ?? 1,
      weight: (json['WEIGHT'] as num?)?.toDouble() ?? 170,
      bpsys: (json['BPSYS'] as num?)?.toDouble() ?? 120,
      bpdias: (json['BPDIAS'] as num?)?.toDouble() ?? 80,
      npiqinf: (json['NPIQINF'] as num?)?.toDouble() ?? 1,
      del: (json['DEL'] as num?)?.toDouble() ?? 0,
      hall: (json['HALL'] as num?)?.toDouble() ?? 0,
      agit: (json['AGIT'] as num?)?.toDouble() ?? 0,
      depd: (json['DEPD'] as num?)?.toDouble() ?? 0,
      anx: (json['ANX'] as num?)?.toDouble() ?? 0,
      elat: (json['ELAT'] as num?)?.toDouble() ?? 0,
      apa: (json['APA'] as num?)?.toDouble() ?? 0,
      disn: (json['DISN'] as num?)?.toDouble() ?? 0,
      irr: (json['IRR'] as num?)?.toDouble() ?? 0,
      mot: (json['MOT'] as num?)?.toDouble() ?? 0,
      nite: (json['NITE'] as num?)?.toDouble() ?? 0,
      app: (json['APP'] as num?)?.toDouble() ?? 0,
      satis: (json['SATIS'] as num?)?.toDouble() ?? 0,
      dropact: (json['DROPACT'] as num?)?.toDouble() ?? 0,
      empty: (json['EMPTY'] as num?)?.toDouble() ?? 0,
      bored: (json['BORED'] as num?)?.toDouble() ?? 0,
      spirits: (json['SPIRITS'] as num?)?.toDouble() ?? 0,
      afraid: (json['AFRAID'] as num?)?.toDouble() ?? 0,
      happy: (json['HAPPY'] as num?)?.toDouble() ?? 0,
      helpless: (json['HELPLESS'] as num?)?.toDouble() ?? 0,
      stayhome: (json['STAYHOME'] as num?)?.toDouble() ?? 0,
      wondrful: (json['WONDRFUL'] as num?)?.toDouble() ?? 0,
      wrthless: (json['WRTHLESS'] as num?)?.toDouble() ?? 0,
      energy: (json['ENERGY'] as num?)?.toDouble() ?? 0,
      hopeless: (json['HOPELESS'] as num?)?.toDouble() ?? 0,
      better: (json['BETTER'] as num?)?.toDouble() ?? 0,
      gds: (json['GDS'] as num?)?.toDouble() ?? 0,
      bills: (json['BILLS'] as num?)?.toDouble() ?? 0,
      taxes: (json['TAXES'] as num?)?.toDouble() ?? 0,
      shopping: (json['SHOPPING'] as num?)?.toDouble() ?? 0,
      games: (json['GAMES'] as num?)?.toDouble() ?? 0,
      stove: (json['STOVE'] as num?)?.toDouble() ?? 0,
      mealprep: (json['MEALPREP'] as num?)?.toDouble() ?? 0,
      events: (json['EVENTS'] as num?)?.toDouble() ?? 0,
      payattn: (json['PAYATTN'] as num?)?.toDouble() ?? 0,
      remdates: (json['REMDATES'] as num?)?.toDouble() ?? 0,
      travel: (json['TRAVEL'] as num?)?.toDouble() ?? 0,
      psp: (json['PSP'] as num?)?.toDouble() ?? 0,
      cort: (json['CORT'] as num?)?.toDouble() ?? 0,
      hunt: (json['HUNT'] as num?)?.toDouble() ?? 0,
      prion: (json['PRION'] as num?)?.toDouble() ?? 0,
      meds: (json['MEDS'] as num?)?.toDouble() ?? 0,
      dysill: (json['DYSILL'] as num?)?.toDouble() ?? 0,
      dep: (json['DEP'] as num?)?.toDouble() ?? 0,
      othpsy: (json['OTHPSY'] as num?)?.toDouble() ?? 0,
      downs: (json['DOWNS'] as num?)?.toDouble() ?? 0,
      park: (json['PARK'] as num?)?.toDouble() ?? 0,
      hyceph: (json['HYCEPH'] as num?)?.toDouble() ?? 0,
      brninj: (json['BRNINJ'] as num?)?.toDouble() ?? 0,
      neop: (json['NEOP'] as num?)?.toDouble() ?? 0,
      cogoth: (json['COGOTH'] as num?)?.toDouble() ?? 0,
      cogoth2: (json['COGOTH2'] as num?)?.toDouble() ?? 0,
      cogoth3: (json['COGOTH3'] as num?)?.toDouble() ?? 0,
      medsEmbeddings: embs,
    );
  }
}
