import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../models/patient_data.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _serverOnline = false;
  bool _checking = true;

  @override
  void initState() {
    super.initState();
    _checkServer();
  }

  Future<void> _checkServer() async {
    final ok = await ApiService.healthCheck();
    if (mounted) setState(() { _serverOnline = ok; _checking = false; });
  }

  void _loadDemoPatient() async {
    final data = await ApiService.getDefaultPatient();
    if (data != null && mounted) {
      final clinical = data['clinical_data'] as Map<String, dynamic>;
      final patient = PatientData.fromJson(clinical);
      Navigator.pushNamed(context, '/clinical-form', arguments: {
        'patient': patient,
        'isDemo': true,
        'patientId': data['patient_id'],
      });
    } else {
      // Offline fallback — use hardcoded severe patient
      final patient = PatientData(
        ageAtVisit: 69, animals: 19, asscmem: 16, srtfree: 27,
        srttotal: 48, tma: 33, tmb: 35, independ: 1, residenc: 1,
        maristat: 1, insex: 2, inrelto: 1, inlivwth: 1, inrely: 0,
        anymeds: 1, weight: 145, bpsys: 2, bpdias: 68, npiqinf: 1,
        agit: 1, depd: 1, apa: 1, irr: 1,
        shopping: 2, games: 8, mealprep: 8, remdates: 1,
      );
      Navigator.pushNamed(context, '/clinical-form', arguments: {
        'patient': patient,
        'isDemo': true,
        'patientId': 'OAS30002',
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(gradient: AppTheme.bgGradient),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                FadeInDown(
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          gradient: AppTheme.primaryGradient,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Icon(Icons.psychology, color: Colors.white, size: 28),
                      ),
                      const SizedBox(width: 16),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          ShaderMask(
                            shaderCallback: (b) => AppTheme.primaryGradient.createShader(b),
                            child: Text('NeuroScan AI',
                                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                    fontSize: 24, fontWeight: FontWeight.w800, color: Colors.white)),
                          ),
                          const SizedBox(height: 2),
                          Text('Alzheimer Sınıflandırma',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                  color: AppTheme.textMuted, fontSize: 13)),
                        ],
                      ),
                      const Spacer(),
                      // Server status
                      FadeIn(
                        delay: const Duration(milliseconds: 500),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: _checking
                                ? AppTheme.bgCardLight
                                : _serverOnline
                                    ? AppTheme.accentGreen.withValues(alpha: 0.15)
                                    : AppTheme.accentPink.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: _checking
                                  ? AppTheme.borderGlass
                                  : _serverOnline
                                      ? AppTheme.accentGreen.withValues(alpha: 0.4)
                                      : AppTheme.accentPink.withValues(alpha: 0.4),
                            ),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (_checking)
                                const SizedBox(
                                    width: 10, height: 10,
                                    child: CircularProgressIndicator(strokeWidth: 1.5))
                              else
                                Icon(
                                  _serverOnline ? Icons.cloud_done : Icons.cloud_off,
                                  size: 14,
                                  color: _serverOnline ? AppTheme.accentGreen : AppTheme.accentPink,
                                ),
                              const SizedBox(width: 6),
                              Text(
                                _checking ? 'Kontrol...' : _serverOnline ? 'Çevrimiçi' : 'Çevrimdışı',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: _checking
                                      ? AppTheme.textMuted
                                      : _serverOnline ? AppTheme.accentGreen : AppTheme.accentPink,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 40),

                // Stats row
                FadeInUp(
                  delay: const Duration(milliseconds: 200),
                  child: Row(
                    children: [
                      _StatChip(icon: Icons.model_training, label: 'ResNet50', value: '3D CNN'),
                      const SizedBox(width: 12),
                      _StatChip(icon: Icons.table_chart, label: 'XGBoost', value: 'Klinik'),
                      const SizedBox(width: 12),
                      _StatChip(icon: Icons.merge_type, label: 'Füzyon', value: 'α=0.35'),
                    ],
                  ),
                ),

                const SizedBox(height: 40),

                // Main action cards
                Expanded(
                  child: Column(
                    children: [
                      // New Analysis Card
                      FadeInUp(
                        delay: const Duration(milliseconds: 400),
                        child: _ActionCard(
                          gradient: AppTheme.primaryGradient,
                          icon: Icons.person_add_alt_1,
                          title: 'Yeni Hasta Analizi',
                          subtitle: 'Klinik verileri girin ve MRI yükleyin',
                          onTap: () {
                            Navigator.pushNamed(context, '/clinical-form', arguments: {
                              'patient': PatientData(),
                              'isDemo': false,
                            });
                          },
                        ),
                      ),
                      const SizedBox(height: 16),
                      // Demo Patient Card
                      FadeInUp(
                        delay: const Duration(milliseconds: 600),
                        child: _ActionCard(
                          gradient: AppTheme.dangerGradient,
                          icon: Icons.science,
                          title: 'Demo Hasta (Ağır Vaka)',
                          subtitle: 'OAS30002 — Moderate/Severe test verisi',
                          onTap: _loadDemoPatient,
                        ),
                      ),
                      const Spacer(),
                      // Info card
                      FadeInUp(
                        delay: const Duration(milliseconds: 800),
                        child: Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(20),
                          decoration: AppTheme.glassDecoration(),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(Icons.info_outline, color: AppTheme.accentTeal, size: 18),
                                  const SizedBox(width: 8),
                                  Text('Geç Füzyon Modeli',
                                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                          color: AppTheme.accentTeal, fontWeight: FontWeight.w700)),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Text(
                                '3D ResNet50 (MRI) + XGBoost (74 klinik özellik) modellerinin '
                                'ağırlıklı kombinasyonu ile 3 sınıf tahmin:\n'
                                'Normal · Hafif · Orta/Ağır',
                                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: AppTheme.textSecondary, height: 1.5),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _StatChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  const _StatChip({required this.icon, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
        decoration: AppTheme.glassDecoration(borderRadius: 16),
        child: Column(
          children: [
            Icon(icon, color: AppTheme.accentTeal, size: 22),
            const SizedBox(height: 6),
            Text(label, style: const TextStyle(color: AppTheme.textMuted, fontSize: 10, fontWeight: FontWeight.w600)),
            const SizedBox(height: 2),
            Text(value, style: const TextStyle(color: AppTheme.textPrimary, fontSize: 12, fontWeight: FontWeight.w700)),
          ],
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final LinearGradient gradient;
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _ActionCard({
    required this.gradient,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: AppTheme.gradientCardDecoration(gradient: gradient),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(icon, color: Colors.white, size: 32),
            ),
            const SizedBox(width: 20),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: const TextStyle(
                          color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 4),
                  Text(subtitle,
                      style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.8), fontSize: 13)),
                ],
              ),
            ),
            Icon(Icons.arrow_forward_ios, color: Colors.white.withValues(alpha: 0.6), size: 18),
          ],
        ),
      ),
    );
  }
}
