import 'dart:convert';
import 'dart:io' show Platform, File;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import '../models/patient_data.dart';

class ApiService {
  static String get _baseUrl {
    if (kIsWeb) {
      return 'http://localhost:8000';
    }
    try {
      if (Platform.isWindows || Platform.isMacOS || Platform.isLinux) {
        return 'http://localhost:8000';
      }
    } catch (_) {}
    return 'http://10.0.2.2:8000'; // Default Android Emulator
  }

  static String get baseUrl => _baseUrl;

  /// Health check
  static Future<bool> healthCheck() async {
    try {
      final res = await http.get(Uri.parse('$_baseUrl/api/health'))
          .timeout(const Duration(seconds: 5));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Get default patient data
  static Future<Map<String, dynamic>?> getDefaultPatient() async {
    try {
      final res = await http.get(Uri.parse('$_baseUrl/api/default-patient'))
          .timeout(const Duration(seconds: 10));
      if (res.statusCode == 200) {
        return jsonDecode(res.body);
      }
    } catch (e) {
      print('Error loading default patient: $e');
    }
    return null;
  }

  /// Predict with clinical data only
  static Future<Map<String, dynamic>?> predictClinicalOnly(PatientData data) async {
    try {
      final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/api/predict/clinical'));
      request.fields['clinical_json'] = jsonEncode(data.toJson());

      final streamedRes = await request.send().timeout(const Duration(seconds: 30));
      final res = await http.Response.fromStream(streamedRes);

      if (res.statusCode == 200) {
        return jsonDecode(res.body);
      }
    } catch (e) {
      print('Error in clinical prediction: $e');
    }
    return null;
  }

  /// Predict with clinical data + MRI file (full fusion)
  static Future<Map<String, dynamic>?> predictFusion({
    required PatientData clinicalData,
    File? mriFile,
    bool useDefaultMri = false,
  }) async {
    try {
      final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/api/predict'));
      request.fields['clinical_json'] = jsonEncode(clinicalData.toJson());
      request.fields['use_default_mri'] = useDefaultMri.toString();

      if (mriFile != null) {
        request.files.add(await http.MultipartFile.fromPath('mri_file', mriFile.path));
      }

      final streamedRes = await request.send().timeout(const Duration(seconds: 120));
      final res = await http.Response.fromStream(streamedRes);

      if (res.statusCode == 200) {
        return jsonDecode(res.body);
      }
    } catch (e) {
      print('Error in fusion prediction: $e');
    }
    return null;
  }
}
