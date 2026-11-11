import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:animate_do/animate_do.dart';
import '../theme/app_theme.dart';
import '../models/patient_data.dart';
import '../services/api_service.dart';

class MriUploadScreen extends StatefulWidget {
  const MriUploadScreen({super.key});

  @override
  State<MriUploadScreen> createState() => _MriUploadScreenState();
}

class _MriUploadScreenState extends State<MriUploadScreen> {
  late PatientData _patient;
  File? _mriFile;
  bool _useDefaultMri = true;
  bool _isLoading = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _patient = ModalRoute.of(context)?.settings.arguments as PatientData? ?? PatientData();
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      allowMultiple: false,
    );
    if (result != null && result.files.single.path != null) {
      setState(() {
        _mriFile = File(result.files.single.path!);
        _useDefaultMri = false;
      });
    }
  }

  Future<void> _runPrediction() async {
    setState(() => _isLoading = true);

    Map<String, dynamic>? result;
    if (_mriFile != null || _useDefaultMri) {
      result = await ApiService.predictFusion(
        clinicalData: _patient,
        mriFile: _mriFile,
        useDefaultMri: _useDefaultMri,
      );
    } else {
      result = await ApiService.predictClinicalOnly(_patient);
    }

    setState(() => _isLoading = false);

    if (result != null && mounted) {
      Navigator.pushNamed(context, '/result', arguments: result);
    } else if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Sunucuya bağlanılamadı. Backend çalıştığından emin olun.'),
          backgroundColor: AppTheme.accentPink,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
      );
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
                // App bar
                Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_ios, color: AppTheme.accentTeal),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const Expanded(
                      child: Text('MRI Görüntüsü',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700)),
                    ),
                    const SizedBox(width: 48),
                  ],
                ),
                const SizedBox(height: 32),

                // Upload area
                FadeInUp(
                  child: GestureDetector(
                    onTap: _pickFile,
                    child: Container(
                      width: double.infinity,
                      height: 200,
                      decoration: BoxDecoration(
                        color: AppTheme.surfaceGlass,
                        borderRadius: BorderRadius.circular(24),
                        border: Border.all(
                          color: _mriFile != null
                              ? AppTheme.accentGreen.withValues(alpha: 0.6)
                              : AppTheme.borderGlass,
                          width: 2,
                        ),
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            _mriFile != null ? Icons.check_circle : Icons.cloud_upload_outlined,
                            size: 56,
                            color: _mriFile != null ? AppTheme.accentGreen : AppTheme.accentTeal,
                          ),
                          const SizedBox(height: 16),
                          Text(
                            _mriFile != null
                                ? _mriFile!.path.split(Platform.pathSeparator).last
                                : 'MRI Dosyası Yükle',
                            style: TextStyle(
                              color: _mriFile != null ? AppTheme.accentGreen : Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _mriFile != null ? 'Değiştirmek için tıklayın' : '.npy veya .nii.gz formatı',
                            style: const TextStyle(color: AppTheme.textMuted, fontSize: 13),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Default MRI option
                FadeInUp(
                  delay: const Duration(milliseconds: 200),
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: AppTheme.glassDecoration(borderRadius: 16),
                    child: Row(
                      children: [
                        Icon(Icons.dataset, color: AppTheme.accentPurple, size: 24),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('Default Test MRI Kullan',
                                  style: TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
                              const Text('OAS30002_MR_d0653.npy',
                                  style: TextStyle(color: AppTheme.textMuted, fontSize: 12)),
                            ],
                          ),
                        ),
                        Switch(
                          value: _useDefaultMri && _mriFile == null,
                          onChanged: (v) => setState(() {
                            _useDefaultMri = v;
                            if (v) _mriFile = null;
                          }),
                          activeColor: AppTheme.accentTeal,
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Clinical-only option
                FadeInUp(
                  delay: const Duration(milliseconds: 300),
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: AppTheme.glassDecoration(borderRadius: 16),
                    child: Row(
                      children: [
                        const Icon(Icons.table_chart, color: AppTheme.accentOrange, size: 24),
                        const SizedBox(width: 14),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('Sadece Klinik Veri',
                                  style: TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
                              Text('MRI olmadan sadece XGBoost',
                                  style: TextStyle(color: AppTheme.textMuted, fontSize: 12)),
                            ],
                          ),
                        ),
                        Switch(
                          value: !_useDefaultMri && _mriFile == null,
                          onChanged: (v) => setState(() {
                            _useDefaultMri = !v;
                            if (v) _mriFile = null;
                          }),
                          activeColor: AppTheme.accentOrange,
                        ),
                      ],
                    ),
                  ),
                ),

                const Spacer(),

                // Submit button
                FadeInUp(
                  delay: const Duration(milliseconds: 500),
                  child: SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      onPressed: _isLoading ? null : _runPrediction,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.accentTeal,
                        foregroundColor: AppTheme.bgDark,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                        elevation: 0,
                      ),
                      child: _isLoading
                          ? const SizedBox(
                              width: 24, height: 24,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5, color: AppTheme.bgDark))
                          : const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.analytics, size: 22),
                                SizedBox(width: 10),
                                Text('Analiz Et',
                                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800)),
                              ],
                            ),
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
