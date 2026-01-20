// ============================================================
// PARCHE: Añadir "Identificar online" al menú de tracks
// Aplicar en local_results_screen.dart
// ============================================================

// =====================================================
// PASO 1: Añadir import de http (si no existe)
// Al inicio del archivo, verificar que exista:
// =====================================================

import 'package:http/http.dart' as http;


// =====================================================
// PASO 2: Añadir la opción en _showTrackOptionsMenu
// Buscar línea ~896 (después de "Cambiar género")
// =====================================================

// BUSCAR:
            if (analysis != null)
              _OptionTile(
                icon: Icons.library_music_rounded,
                label: 'Cambiar género',
                color: Colors.orangeAccent,
                onTap: () {
                  Navigator.pop(context);
                  _showGenreEditDialog(track, analysis);
                },
              ),
            _OptionTile(
              icon: Icons.delete_outline_rounded,
              label: 'Eliminar de biblioteca',

// REEMPLAZAR POR:
            if (analysis != null)
              _OptionTile(
                icon: Icons.library_music_rounded,
                label: 'Cambiar género',
                color: Colors.orangeAccent,
                onTap: () {
                  Navigator.pop(context);
                  _showGenreEditDialog(track, analysis);
                },
              ),
            _OptionTile(
              icon: Icons.travel_explore_rounded,
              label: 'Identificar online',
              subtitle: 'Busca artista y título automáticamente',
              color: Colors.cyanAccent,
              onTap: () {
                Navigator.pop(context);
                _identifyTrackOnline(track);
              },
            ),
            _OptionTile(
              icon: Icons.delete_outline_rounded,
              label: 'Eliminar de biblioteca',


// =====================================================
// PASO 3: Añadir el método _identifyTrackOnline
// Añadir después de _removeTrack (~línea 825)
// =====================================================

  Future<void> _identifyTrackOnline(TrackMeta track) async {
    if (track.path == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No se puede identificar: archivo no encontrado'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    // Mostrar diálogo de progreso
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey.shade900,
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(color: Colors.cyanAccent),
            const SizedBox(height: 20),
            const Text(
              'Identificando canción...',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              'Analizando audio para buscar coincidencias',
              style: TextStyle(fontSize: 12, color: Colors.white.withOpacity(0.6)),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );

    try {
      final file = File(track.path!);
      if (!await file.exists()) {
        if (mounted) Navigator.pop(context);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Archivo no encontrado'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }

      // Enviar al backend
      final uri = Uri.parse('${ApiConfig.backendUrl}/identify');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', track.path!));

      final response = await request.send().timeout(const Duration(seconds: 60));
      final responseBody = await response.stream.bytesToString();

      if (mounted) Navigator.pop(context); // Cerrar diálogo de progreso

      if (response.statusCode != 200) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error del servidor: ${response.statusCode}'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }

      final data = jsonDecode(responseBody);

      if (data['status'] == 'found') {
        final artist = data['artist'] ?? '';
        final title = data['title'] ?? '';
        final album = data['album'];
        final label = data['label'];

        // Mostrar resultado
        if (mounted) {
          _showIdentificationResult(track, artist, title, album, label, data);
        }
      } else if (data['status'] == 'not_found') {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text('No se pudo identificar la canción'),
              backgroundColor: Colors.orange.shade700,
              action: SnackBarAction(
                label: 'Editar manual',
                textColor: Colors.white,
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => TrackEditScreen(track: track)),
                  );
                },
              ),
            ),
          );
        }
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(data['message'] ?? 'Error desconocido'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) Navigator.pop(context);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _showIdentificationResult(
    TrackMeta track,
    String artist,
    String title,
    String? album,
    String? label,
    Map<String, dynamic> fullData,
  ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey.shade900,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.cyanAccent.withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.check_circle_rounded, color: Colors.cyanAccent, size: 24),
            ),
            const SizedBox(width: 12),
            const Expanded(
              child: Text('¡Canción identificada!', style: TextStyle(fontSize: 18)),
            ),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Título
            Text(
              title,
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              artist,
              style: TextStyle(fontSize: 16, color: Colors.white.withOpacity(0.7)),
            ),
            
            if (album != null || label != null) ...[
              const SizedBox(height: 16),
              if (album != null)
                _infoRow(Icons.album_rounded, 'Álbum', album),
              if (label != null)
                _infoRow(Icons.business_rounded, 'Sello', label),
            ],
            
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.cyanAccent.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.cyanAccent.withOpacity(0.3)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info_outline_rounded, color: Colors.cyanAccent, size: 18),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      '¿Aplicar estos datos al track?',
                      style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.8)),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancelar', style: TextStyle(color: Colors.white.withOpacity(0.6))),
          ),
          ElevatedButton.icon(
            onPressed: () {
              Navigator.pop(context);
              _applyIdentificationData(track, artist, title, album, label);
            },
            icon: const Icon(Icons.check_rounded, size: 18),
            label: const Text('Aplicar'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.cyanAccent,
              foregroundColor: Colors.black,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _infoRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Icon(icon, size: 16, color: Colors.white.withOpacity(0.5)),
          const SizedBox(width: 8),
          Text(
            '$label: ',
            style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.5)),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontSize: 13),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _applyIdentificationData(
    TrackMeta track,
    String artist,
    String title,
    String? album,
    String? label,
  ) async {
    // Guardar como override
    final override = trackOverrides[track.id] ?? TrackOverride(trackId: track.id);
    override.artist = artist;
    override.title = title;
    trackOverrides[track.id] = override;
    
    await saveOverrides();
    _rebuildTracksFromCache();
    
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('✓ Actualizado: $artist - $title'),
          backgroundColor: Colors.green.shade700,
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }


// =====================================================
// PASO 4: Actualizar _OptionTile para soportar subtitle
// Buscar el widget _OptionTile (al final del archivo)
// =====================================================

// Si _OptionTile no tiene parámetro subtitle, actualizar así:

class _OptionTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String? subtitle;
  final Color color;
  final VoidCallback onTap;

  const _OptionTile({
    required this.icon,
    required this.label,
    this.subtitle,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: color, size: 22),
      ),
      title: Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
      subtitle: subtitle != null 
          ? Text(subtitle!, style: TextStyle(fontSize: 12, color: Colors.white.withOpacity(0.5)))
          : null,
      trailing: Icon(Icons.chevron_right_rounded, color: Colors.white.withOpacity(0.3)),
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    );
  }
}
