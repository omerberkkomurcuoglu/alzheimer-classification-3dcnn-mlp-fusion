import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../models/patient_data.dart';

class ClinicalFormScreen extends StatefulWidget {
  const ClinicalFormScreen({super.key});

  @override
  State<ClinicalFormScreen> createState() => _ClinicalFormScreenState();
}

class _ClinicalFormScreenState extends State<ClinicalFormScreen> {
  late PatientData _patient;
  bool _isDemo = false;
  String? _patientId;
  int _currentSection = 0;

  final _formKey = GlobalKey<FormState>();

  // Controllers for each field
  final Map<String, TextEditingController> _controllers = {};

  static const List<String> _sectionTitles = [
    'Demografik Bilgiler',
    'Bilişsel Testler',
    'Fiziksel Ölçümler',
    'NPI-Q Semptomlar',
    'GDS Depresyon',
    'FAQ Fonksiyonel',
    'Tıbbi Geçmiş',
  ];

  static const List<IconData> _sectionIcons = [
    Icons.person,
    Icons.psychology_alt,
    Icons.monitor_heart,
    Icons.mood,
    Icons.sentiment_very_dissatisfied,
    Icons.accessibility_new,
    Icons.medical_information,
  ];

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>?;
    if (args != null && _controllers.isEmpty) {
      _patient = args['patient'] as PatientData;
      _isDemo = args['isDemo'] as bool? ?? false;
      _patientId = args['patientId'] as String?;
      _initControllers();
    }
  }

  void _initControllers() {
    final json = _patient.toJson();
    // Only create controllers for user-editable fields (not med embeddings)
    for (final entry in json.entries) {
      if (!entry.key.startsWith('meds_')) {
        _controllers[entry.key] = TextEditingController(text: entry.value.toString());
      }
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) { c.dispose(); }
    super.dispose();
  }

  void _updatePatientFromControllers() {
    final Map<String, dynamic> json = {};
    for (final entry in _controllers.entries) {
      json[entry.key] = double.tryParse(entry.value.text) ?? 0.0;
    }
    // Keep original med embeddings
    final origJson = _patient.toJson();
    for (final key in origJson.keys) {
      if (key.startsWith('meds_')) {
        json[key] = origJson[key];
      }
    }
    _patient = PatientData.fromJson(json);
  }

  List<_FieldDef> _getFieldsForSection(int section) {
    switch (section) {
      case 0: return [
        _FieldDef('age at visit', 'Yaş', 'Hasta yaşı', Icons.cake),
        _FieldDef('INDEPEND', 'Bağımsızlık', '1=Bağımsız, 2=Yarı, 3=Bağımlı', Icons.accessibility),
        _FieldDef('RESIDENC', 'Yaşam Yeri', '1=Ev, 2=Bakımevi', Icons.home),
        _FieldDef('MARISTAT', 'Medeni Hal', '1=Evli, 2=Bekar...', Icons.people),
        _FieldDef('INSEX', 'Cinsiyet', '1=Erkek, 2=Kadın', Icons.wc),
        _FieldDef('INRELTO', 'Bilgi Veren Yakınlık', '1=Eş, 2=Çocuk...', Icons.family_restroom),
        _FieldDef('INLIVWTH', 'Birlikte Yaşama', '0=Hayır, 1=Evet', Icons.group),
        _FieldDef('INRELY', 'Bilgi Güvenilirliği', '0=Güvenilir', Icons.verified),
        _FieldDef('anymeds', 'İlaç Kullanımı', '0=Hayır, 1=Evet', Icons.medication),
      ];
      case 1: return [
        _FieldDef('ANIMALS', 'Hayvan İsimlendirme', '1 dk hayvan sayma (normal ≥18)', Icons.pets),
        _FieldDef('asscmem', 'İlişkisel Bellek', 'Bellek skoru (normal ≥14)', Icons.memory),
        _FieldDef('srtfree', 'Serbest Hatırlama', 'Kelime hatırlama (normal ≥30)', Icons.record_voice_over),
        _FieldDef('srttotal', 'Toplam Hatırlama', 'Toplam skor (max 48)', Icons.assessment),
        _FieldDef('tma', 'Trail Making A', 'Saniye (normal <30s)', Icons.timer),
        _FieldDef('tmb', 'Trail Making B', 'Saniye (normal <60s)', Icons.timer_10),
      ];
      case 2: return [
        _FieldDef('WEIGHT', 'Kilo', 'lbs cinsinden', Icons.monitor_weight),
        _FieldDef('BPSYS', 'Sistolik KB', 'mmHg', Icons.favorite),
        _FieldDef('BPDIAS', 'Diastolik KB', 'mmHg', Icons.heart_broken),
      ];
      case 3: return [
        _FieldDef('NPIQINF', 'NPI Bilgi Kaynağı', '1=Eş, 2=Çocuk, 3=Bakıcı', Icons.info),
        _FieldDef('DEL', 'Sanrılar', '0=Yok, 1-3=Şiddet', Icons.visibility_off),
        _FieldDef('HALL', 'Halüsinasyonlar', '0=Yok, 1-3=Şiddet', Icons.blur_on),
        _FieldDef('AGIT', 'Ajitasyon', '0=Yok, 1-3=Şiddet', Icons.flash_on),
        _FieldDef('DEPD', 'Depresyon', '0=Yok, 1-3=Şiddet', Icons.sentiment_very_dissatisfied),
        _FieldDef('ANX', 'Anksiyete', '0=Yok, 1-3=Şiddet', Icons.warning_amber),
        _FieldDef('ELAT', 'Öfori', '0=Yok, 1-3=Şiddet', Icons.mood),
        _FieldDef('APA', 'Apati', '0=Yok, 1-3=Şiddet', Icons.do_not_disturb_on),
        _FieldDef('DISN', 'Disinhibisyon', '0=Yok, 1-3=Şiddet', Icons.block),
        _FieldDef('IRR', 'İrritabilite', '0=Yok, 1-3=Şiddet', Icons.local_fire_department),
        _FieldDef('MOT', 'Motor Bozukluk', '0=Yok, 1-3=Şiddet', Icons.directions_walk),
        _FieldDef('NITE', 'Gece Davranışları', '0=Yok, 1-3=Şiddet', Icons.nightlight),
        _FieldDef('APP', 'İştah Bozukluğu', '0=Yok, 1-3=Şiddet', Icons.restaurant),
      ];
      case 4: return [
        _FieldDef('SATIS', 'Yaşam Memnuniyeti', '0=Evet, 1=Hayır', Icons.thumb_up),
        _FieldDef('DROPACT', 'Aktivite Bırakma', '0=Hayır, 1=Evet', Icons.cancel),
        _FieldDef('EMPTY', 'Boşluk Hissi', '0=Hayır, 1=Evet', Icons.circle_outlined),
        _FieldDef('BORED', 'Sıkılma', '0=Hayır, 1=Evet', Icons.sentiment_neutral),
        _FieldDef('SPIRITS', 'Moral Düşüklüğü', '0=Hayır, 1=Evet', Icons.trending_down),
        _FieldDef('AFRAID', 'Korku', '0=Hayır, 1=Evet', Icons.shield),
        _FieldDef('HAPPY', 'Mutluluk', '0=Evet, 1=Hayır', Icons.sentiment_satisfied),
        _FieldDef('HELPLESS', 'Çaresizlik', '0=Hayır, 1=Evet', Icons.help_outline),
        _FieldDef('STAYHOME', 'Evde Kalma', '0=Hayır, 1=Evet', Icons.home_work),
        _FieldDef('WONDRFUL', 'Harika Hissetme', '0=Evet, 1=Hayır', Icons.star),
        _FieldDef('WRTHLESS', 'Değersizlik', '0=Hayır, 1=Evet', Icons.do_not_disturb),
        _FieldDef('ENERGY', 'Enerji', '0=Yeterli, 1=Yetersiz', Icons.bolt),
        _FieldDef('HOPELESS', 'Umutsuzluk', '0=Hayır, 1=Evet', Icons.cloud),
        _FieldDef('BETTER', 'Daha İyi Hissetme', '0=Evet, 1=Hayır', Icons.trending_up),
        _FieldDef('GDS', 'GDS Toplam', 'Toplam puan (0-15)', Icons.summarize),
      ];
      case 5: return [
        _FieldDef('BILLS', 'Fatura Ödeme', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.receipt_long),
        _FieldDef('TAXES', 'Vergi İşleri', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.account_balance),
        _FieldDef('SHOPPING', 'Alışveriş', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.shopping_cart),
        _FieldDef('GAMES', 'Oyun/Hobi', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.sports_esports),
        _FieldDef('STOVE', 'Ocak Kullanımı', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.local_fire_department),
        _FieldDef('MEALPREP', 'Yemek Hazırlama', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.restaurant_menu),
        _FieldDef('EVENTS', 'Güncel Olaylar', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.newspaper),
        _FieldDef('PAYATTN', 'Dikkat', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.visibility),
        _FieldDef('REMDATES', 'Tarih Hatırlama', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.calendar_today),
        _FieldDef('TRAVEL', 'Seyahat', '0=Normal, 1-3=Zorluk, 8=N/A', Icons.flight),
      ];
      case 6: return [
        _FieldDef('PSP', 'PSP', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('CORT', 'Kortikal', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('HUNT', 'Huntington', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('PRION', 'Prion', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('DYSILL', 'Dismetabolik', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('DEP', 'Depresyon Hx', '0=Hayır, 1=Evet', Icons.psychology),
        _FieldDef('OTHPSY', 'Diğer Psikiyatrik', '0=Hayır, 1=Evet', Icons.psychology),
        _FieldDef('DOWNS', 'Down Sendromu', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('PARK', 'Parkinson', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('HYCEPH', 'Hidrosefali', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('BRNINJ', 'Beyin Hasarı', '0=Hayır, 1=Evet', Icons.medical_services),
        _FieldDef('NEOP', 'Neoplazm', '0=Hayır, 1=Evet', Icons.medical_services),
      ];
      default: return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    final fields = _getFieldsForSection(_currentSection);

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(gradient: AppTheme.bgGradient),
        child: SafeArea(
          child: Column(
            children: [
              // App bar
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_ios, color: AppTheme.accentTeal),
                      onPressed: () => Navigator.pop(context),
                    ),
                    Expanded(
                      child: Column(
                        children: [
                          Text(
                            _isDemo ? 'Demo Hasta ($_patientId)' : 'Hasta Bilgileri',
                            style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            '${_currentSection + 1} / ${_sectionTitles.length}',
                            style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                    TextButton(
                      onPressed: () {
                        _updatePatientFromControllers();
                        Navigator.pushNamed(context, '/mri-upload', arguments: _patient);
                      },
                      child: const Text('MRI →', style: TextStyle(color: AppTheme.accentTeal, fontWeight: FontWeight.w700)),
                    ),
                  ],
                ),
              ),

              // Section progress
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: (_currentSection + 1) / _sectionTitles.length,
                    backgroundColor: AppTheme.bgCardLight,
                    valueColor: const AlwaysStoppedAnimation(AppTheme.accentTeal),
                    minHeight: 3,
                  ),
                ),
              ),

              // Section selector
              SizedBox(
                height: 50,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  itemCount: _sectionTitles.length,
                  itemBuilder: (context, i) {
                    final selected = i == _currentSection;
                    return GestureDetector(
                      onTap: () => setState(() => _currentSection = i),
                      child: Container(
                        margin: const EdgeInsets.only(right: 8),
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                        decoration: BoxDecoration(
                          color: selected ? AppTheme.accentTeal.withValues(alpha: 0.15) : Colors.transparent,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: selected ? AppTheme.accentTeal : AppTheme.borderGlass,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(_sectionIcons[i], size: 14,
                                color: selected ? AppTheme.accentTeal : AppTheme.textMuted),
                            const SizedBox(width: 6),
                            Text(_sectionTitles[i],
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                                  color: selected ? AppTheme.accentTeal : AppTheme.textSecondary,
                                )),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),

              // Form fields
              Expanded(
                child: Form(
                  key: _formKey,
                  child: ListView.builder(
                    padding: const EdgeInsets.all(20),
                    itemCount: fields.length,
                    itemBuilder: (context, i) {
                      final f = fields[i];
                      final ctrl = _controllers[f.key];
                      if (ctrl == null) return const SizedBox.shrink();
                      return Container(
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.all(16),
                        decoration: AppTheme.glassDecoration(borderRadius: 16),
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: AppTheme.accentTeal.withValues(alpha: 0.1),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Icon(f.icon, color: AppTheme.accentTeal, size: 20),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(f.label,
                                      style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
                                  Text(f.hint,
                                      style: const TextStyle(color: AppTheme.textMuted, fontSize: 11)),
                                ],
                              ),
                            ),
                            const SizedBox(width: 12),
                            SizedBox(
                              width: 80,
                              child: TextFormField(
                                controller: ctrl,
                                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
                                decoration: InputDecoration(
                                  isDense: true,
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
                                  filled: true,
                                  fillColor: AppTheme.bgDark,
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(10),
                                    borderSide: const BorderSide(color: AppTheme.borderGlass),
                                  ),
                                  enabledBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(10),
                                    borderSide: const BorderSide(color: AppTheme.borderGlass),
                                  ),
                                  focusedBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(10),
                                    borderSide: const BorderSide(color: AppTheme.accentTeal, width: 2),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ),

              // Navigation buttons
              Padding(
                padding: const EdgeInsets.all(20),
                child: Row(
                  children: [
                    if (_currentSection > 0)
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () => setState(() => _currentSection--),
                          icon: const Icon(Icons.arrow_back, size: 18),
                          label: const Text('Önceki'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: AppTheme.textSecondary,
                            side: const BorderSide(color: AppTheme.borderGlass),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                          ),
                        ),
                      ),
                    if (_currentSection > 0) const SizedBox(width: 12),
                    Expanded(
                      flex: 2,
                      child: ElevatedButton.icon(
                        onPressed: () {
                          if (_currentSection < _sectionTitles.length - 1) {
                            setState(() => _currentSection++);
                          } else {
                            _updatePatientFromControllers();
                            Navigator.pushNamed(context, '/mri-upload', arguments: _patient);
                          }
                        },
                        icon: Icon(
                          _currentSection < _sectionTitles.length - 1
                              ? Icons.arrow_forward
                              : Icons.upload_file,
                          size: 18,
                        ),
                        label: Text(
                          _currentSection < _sectionTitles.length - 1 ? 'Sonraki' : 'MRI Yükleme →',
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FieldDef {
  final String key;
  final String label;
  final String hint;
  final IconData icon;
  const _FieldDef(this.key, this.label, this.hint, this.icon);
}
