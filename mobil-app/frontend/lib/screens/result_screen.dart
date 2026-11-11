import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:fl_chart/fl_chart.dart';
import '../theme/app_theme.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final result = ModalRoute.of(context)?.settings.arguments as Map<String, dynamic>? ?? {};

    final predictedClass = result['predicted_class'] as int? ?? 0;
    final predictedLabel = result['predicted_label'] as String? ?? 'Unknown';
    final description = result['description'] as String? ?? '';
    final confidence = (result['confidence'] as num?)?.toDouble() ?? 0.0;
    final mode = result['mode'] as String? ?? 'unknown';
    final probs = (result['probabilities'] as Map<String, dynamic>?) ?? {};
    final mriProbs = result['mri_probabilities'] as Map<String, dynamic>?;
    final clinicalProbs = result['clinical_probabilities'] as Map<String, dynamic>?;

    final classColor = AppTheme.classColor(predictedClass);
    final classNames = ['Normal', 'Mild', 'Moderate/Severe'];

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(gradient: AppTheme.bgGradient),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                // Header
                Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_ios, color: AppTheme.accentTeal),
                      onPressed: () => Navigator.popUntil(context, ModalRoute.withName('/home')),
                    ),
                    const Expanded(
                      child: Text('Analiz Sonucu',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700)),
                    ),
                    const SizedBox(width: 48),
                  ],
                ),

                const SizedBox(height: 24),

                // Main result card
                FadeInUp(
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(28),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [classColor.withValues(alpha: 0.2), classColor.withValues(alpha: 0.05)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(color: classColor.withValues(alpha: 0.4)),
                    ),
                    child: Column(
                      children: [
                        // Icon
                        Container(
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            color: classColor.withValues(alpha: 0.15),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            predictedClass == 0
                                ? Icons.check_circle
                                : predictedClass == 1
                                    ? Icons.warning_amber
                                    : Icons.error,
                            size: 48,
                            color: classColor,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(predictedLabel,
                            style: TextStyle(
                                color: classColor, fontSize: 28, fontWeight: FontWeight.w800)),
                        const SizedBox(height: 8),
                        Text(description,
                            textAlign: TextAlign.center,
                            style: const TextStyle(color: AppTheme.textSecondary, fontSize: 14, height: 1.5)),
                        const SizedBox(height: 20),
                        // Confidence
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text('Güven: ', style: TextStyle(color: AppTheme.textMuted, fontSize: 14)),
                            Text('${(confidence * 100).toStringAsFixed(1)}%',
                                style: TextStyle(
                                    color: classColor, fontSize: 22, fontWeight: FontWeight.w800)),
                          ],
                        ),
                        const SizedBox(height: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
                          decoration: BoxDecoration(
                            color: AppTheme.surfaceGlass,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            mode == 'fusion' ? '🔀 Geç Füzyon (ResNet50 + XGBoost)'
                                : mode == 'clinical_only' ? '📊 Sadece Klinik (XGBoost)'
                                : '🧠 Sadece MRI (ResNet50)',
                            style: const TextStyle(color: AppTheme.textSecondary, fontSize: 12),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // Probability bar chart
                FadeInUp(
                  delay: const Duration(milliseconds: 200),
                  child: Container(
                    padding: const EdgeInsets.all(20),
                    decoration: AppTheme.glassDecoration(),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.bar_chart, color: AppTheme.accentTeal, size: 20),
                            SizedBox(width: 8),
                            Text('Füzyon Olasılıkları',
                                style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
                          ],
                        ),
                        const SizedBox(height: 20),
                        ...classNames.asMap().entries.map((entry) {
                          final i = entry.key;
                          final name = entry.value;
                          final prob = (probs[name] as num?)?.toDouble() ?? 0;
                          final color = AppTheme.classColor(i);
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 14),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(name, style: TextStyle(color: color, fontSize: 13, fontWeight: FontWeight.w600)),
                                    Text('${(prob * 100).toStringAsFixed(1)}%',
                                        style: TextStyle(color: color, fontSize: 13, fontWeight: FontWeight.w800)),
                                  ],
                                ),
                                const SizedBox(height: 6),
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(6),
                                  child: LinearProgressIndicator(
                                    value: prob,
                                    backgroundColor: color.withValues(alpha: 0.1),
                                    valueColor: AlwaysStoppedAnimation(color),
                                    minHeight: 10,
                                  ),
                                ),
                              ],
                            ),
                          );
                        }),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Pie chart
                if (probs.isNotEmpty)
                  FadeInUp(
                    delay: const Duration(milliseconds: 400),
                    child: Container(
                      padding: const EdgeInsets.all(20),
                      decoration: AppTheme.glassDecoration(),
                      child: Column(
                        children: [
                          const Text('Sınıf Dağılımı',
                              style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
                          const SizedBox(height: 16),
                          SizedBox(
                            height: 180,
                            child: PieChart(
                              PieChartData(
                                sectionsSpace: 3,
                                centerSpaceRadius: 40,
                                sections: classNames.asMap().entries.map((entry) {
                                  final i = entry.key;
                                  final name = entry.value;
                                  final prob = (probs[name] as num?)?.toDouble() ?? 0;
                                  final color = AppTheme.classColor(i);
                                  return PieChartSectionData(
                                    value: prob * 100,
                                    title: '${(prob * 100).toStringAsFixed(0)}%',
                                    color: color,
                                    radius: i == predictedClass ? 55 : 45,
                                    titleStyle: TextStyle(
                                      color: Colors.white,
                                      fontSize: i == predictedClass ? 14 : 11,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  );
                                }).toList(),
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            children: classNames.asMap().entries.map((e) {
                              return Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Container(width: 10, height: 10,
                                      decoration: BoxDecoration(
                                          color: AppTheme.classColor(e.key),
                                          shape: BoxShape.circle)),
                                  const SizedBox(width: 4),
                                  Text(e.value,
                                      style: const TextStyle(color: AppTheme.textSecondary, fontSize: 11)),
                                ],
                              );
                            }).toList(),
                          ),
                        ],
                      ),
                    ),
                  ),

                const SizedBox(height: 16),

                // Per-modality details
                if (mriProbs != null || clinicalProbs != null)
                  FadeInUp(
                    delay: const Duration(milliseconds: 600),
                    child: Container(
                      padding: const EdgeInsets.all(20),
                      decoration: AppTheme.glassDecoration(),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('Modalite Detayları',
                              style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700)),
                          const SizedBox(height: 14),
                          if (mriProbs != null) ...[
                            const Text('🧠 ResNet50 (MRI)',
                                style: TextStyle(color: AppTheme.accentPurple, fontSize: 13, fontWeight: FontWeight.w600)),
                            const SizedBox(height: 6),
                            ...classNames.map((name) {
                              final p = (mriProbs[name] as num?)?.toDouble() ?? 0;
                              return Padding(
                                padding: const EdgeInsets.only(bottom: 4),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text('  $name', style: const TextStyle(color: AppTheme.textSecondary, fontSize: 12)),
                                    Text('${(p * 100).toStringAsFixed(1)}%',
                                        style: const TextStyle(color: AppTheme.textPrimary, fontSize: 12, fontWeight: FontWeight.w600)),
                                  ],
                                ),
                              );
                            }),
                            const SizedBox(height: 12),
                          ],
                          if (clinicalProbs != null) ...[
                            const Text('📊 XGBoost (Klinik)',
                                style: TextStyle(color: AppTheme.accentOrange, fontSize: 13, fontWeight: FontWeight.w600)),
                            const SizedBox(height: 6),
                            ...classNames.map((name) {
                              final p = (clinicalProbs[name] as num?)?.toDouble() ?? 0;
                              return Padding(
                                padding: const EdgeInsets.only(bottom: 4),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text('  $name', style: const TextStyle(color: AppTheme.textSecondary, fontSize: 12)),
                                    Text('${(p * 100).toStringAsFixed(1)}%',
                                        style: const TextStyle(color: AppTheme.textPrimary, fontSize: 12, fontWeight: FontWeight.w600)),
                                  ],
                                ),
                              );
                            }),
                          ],
                        ],
                      ),
                    ),
                  ),

                const SizedBox(height: 24),

                // Back to home
                SizedBox(
                  width: double.infinity,
                  height: 52,
                  child: OutlinedButton.icon(
                    onPressed: () => Navigator.popUntil(context, ModalRoute.withName('/home')),
                    icon: const Icon(Icons.home, size: 20),
                    label: const Text('Ana Sayfaya Dön'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppTheme.accentTeal,
                      side: const BorderSide(color: AppTheme.accentTeal),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
