#!/usr/bin/env python3
"""
Generate test biometric data files for the Hansken Biometric Scanner Plugin.
Run this on the Hansken VM, then zip the output and upload to Hansken.
"""

import pickle
import json
import numpy as np
import struct
import os

OUTPUT_DIR = '/tmp/biometric_test_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_face_encodings_pkl():
    """Standard face recognition pickle with encodings + names (dlib style)."""
    data = {
        'encodings': [np.random.randn(128).astype(np.float32) for _ in range(5)],
        'names': ['person_a', 'person_b', 'person_c', 'person_d', 'person_e'],
    }
    with open(f'{OUTPUT_DIR}/face_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Created face_encodings.pkl (128-dim dlib style)')


def create_face_embeddings_facenet_pkl():
    """FaceNet-style embeddings with 512-dim vectors."""
    data = {
        'embeddings': [np.random.randn(512).astype(np.float32) for _ in range(3)],
        'labels': ['alice', 'bob', 'charlie'],
        'model': 'facenet_v1',
    }
    with open(f'{OUTPUT_DIR}/facenet_embeddings.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Created facenet_embeddings.pkl (512-dim FaceNet)')


def create_generic_pkl():
    """Generic-named pickle with biometric content (tests content analysis)."""
    data = {
        'encodings': [np.random.randn(128).astype(np.float32)],
        'names': ['unknown_person'],
    }
    with open(f'{OUTPUT_DIR}/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Created data.pkl (generic name, 128-dim — tests content analysis)')


def create_arcface_pkl():
    """ArcFace-style with 512-dim embeddings."""
    data = {
        'face_embeddings': [np.random.randn(512).astype(np.float32) for _ in range(10)],
        'identities': [f'subject_{i}' for i in range(10)],
    }
    with open(f'{OUTPUT_DIR}/arcface_model.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Created arcface_model.pkl (512-dim ArcFace)')


def create_voice_speaker_pkl():
    """Voice/speaker recognition embeddings (x-vector style)."""
    data = {
        'voiceprint': [np.random.randn(512).astype(np.float32) for _ in range(2)],
        'speaker_ids': ['speaker_1', 'speaker_2'],
    }
    with open(f'{OUTPUT_DIR}/voice_speaker.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Created voice_speaker.pkl (speaker recognition)')


def create_numpy_embeddings_npy():
    """NumPy array of face embeddings (FaceNet output format)."""
    embeddings = np.random.randn(20, 128).astype(np.float32)
    np.save(f'{OUTPUT_DIR}/face_embeddings.npy', embeddings)
    print('Created face_embeddings.npy (20x128 numpy array)')


def create_numpy_faces_npz():
    """NumPy compressed archive with embeddings + labels."""
    np.savez(
        f'{OUTPUT_DIR}/face_data.npz',
        embeddings=np.random.randn(15, 512).astype(np.float32),
        labels=np.array(['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15']),
    )
    print('Created face_data.npz (15x512 numpy compressed)')


def create_json_embeddings():
    """JSON file with face embedding data."""
    data = {
        'embeddings': [
            [float(x) for x in np.random.randn(128)],
            [float(x) for x in np.random.randn(128)],
        ],
        'names': ['person_x', 'person_y'],
        'model': 'dlib_face_recognition_resnet_v1',
    }
    with open(f'{OUTPUT_DIR}/face_encodings.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('Created face_encodings.json (128-dim JSON embeddings)')


def create_voice_json():
    """JSON file with voice biometric data."""
    data = {
        'voice_samples': [
            {'speaker': 'subject_1', 'voiceprint': [float(x) for x in np.random.randn(256)]},
            {'speaker': 'subject_2', 'voiceprint': [float(x) for x in np.random.randn(256)]},
        ],
        'model': 'xvector_voice',
    }
    with open(f'{OUTPUT_DIR}/voice_samples.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('Created voice_samples.json (256-dim x-vector)')


def create_fake_dlib_dat():
    """Fake dlib shape predictor .dat file (random binary)."""
    with open(f'{OUTPUT_DIR}/shape_predictor_68_face_landmarks.dat', 'wb') as f:
        f.write(os.urandom(1024))
    print('Created shape_predictor_68_face_landmarks.dat (fake dlib model)')


def create_non_biometric_pkl():
    """Non-biometric pickle (should NOT trigger detection)."""
    data = {
        'config': {'lr': 0.001, 'epochs': 100},
        'weights': [0.1, 0.2, 0.3],
    }
    with open(f'{OUTPUT_DIR}/config.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Created config.pkl (non-biometric — should NOT match)')


def main():
    print(f'Creating test biometric data in {OUTPUT_DIR}/')
    print('-' * 60)

    create_face_encodings_pkl()
    create_face_embeddings_facenet_pkl()
    create_generic_pkl()
    create_arcface_pkl()
    create_voice_speaker_pkl()
    create_numpy_embeddings_npy()
    create_numpy_faces_npz()
    create_json_embeddings()
    create_voice_json()
    create_fake_dlib_dat()
    create_non_biometric_pkl()

    print('-' * 60)
    print(f'Done! {len(os.listdir(OUTPUT_DIR))} files created in {OUTPUT_DIR}/')
    print()
    print('To create a zip for Hansken upload:')
    print(f'  cd /tmp && zip -r biometric_test_data.zip biometric_test_data/')
    print()
    print('Files that SHOULD trigger detection:')
    print('  face_encodings.pkl      — 128-dim dlib style')
    print('  facenet_embeddings.pkl  — 512-dim FaceNet')
    print('  data.pkl                — generic name, content analysis needed')
    print('  arcface_model.pkl       — 512-dim ArcFace')
    print('  voice_speaker.pkl       — voice biometric')
    print('  face_embeddings.npy     — numpy embeddings')
    print('  face_data.npz           — numpy compressed')
    print('  face_encodings.json     — JSON embeddings')
    print('  voice_samples.json      — JSON voice biometric')
    print('  shape_predictor_68_face_landmarks.dat — dlib model')
    print()
    print('Files that should NOT trigger detection:')
    print('  config.pkl              — non-biometric pickle')


if __name__ == '__main__':
    main()