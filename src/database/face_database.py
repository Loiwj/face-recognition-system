"""
Database module để lưu trữ embeddings và metadata
"""

import sqlite3
import json
import pickle
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import hashlib


@dataclass
class PersonRecord:
    """Record cho một người trong database"""
    id: int
    name: str
    embeddings: List[np.ndarray]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    image_paths: List[str]
    confidence_scores: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'image_paths': self.image_paths,
            'confidence_scores': self.confidence_scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonRecord':
        """Tạo từ dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            embeddings=[np.array(emb) for emb in data['embeddings']],
            metadata=data['metadata'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            image_paths=data['image_paths'],
            confidence_scores=data['confidence_scores']
        )


class FaceDatabase:
    """Class chính để quản lý database khuôn mặt"""
    
    def __init__(self, db_path: str = "face_database.db", 
                 backup_path: Optional[str] = None):
        self.db_path = db_path
        self.backup_path = backup_path or f"{db_path}.backup"
        self.logger = logging.getLogger(__name__)
        
        # Tạo database nếu chưa có
        self._create_database()
    
    def _create_database(self):
        """Tạo database và tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tạo table persons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    embeddings BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_paths TEXT,
                    confidence_scores TEXT
                )
            ''')
            
            # Tạo table embeddings (để lưu từng embedding riêng biệt)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    embedding BLOB,
                    confidence REAL,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            # Tạo index
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_name ON persons (name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embedding_person ON embeddings (person_id)')
            
            conn.commit()
    
    def add_person(self, name: str, embeddings: List[np.ndarray], 
                   image_paths: List[str] = None,
                   confidence_scores: List[float] = None,
                   metadata: Dict[str, Any] = None) -> int:
        """
        Thêm người mới vào database
        
        Args:
            name: Tên người
            embeddings: Danh sách embedding vectors
            image_paths: Đường dẫn ảnh
            confidence_scores: Confidence scores
            metadata: Metadata bổ sung
            
        Returns:
            ID của người vừa thêm
        """
        if not embeddings:
            raise ValueError("Embeddings không được trống")
        
        if image_paths is None:
            image_paths = []
        if confidence_scores is None:
            confidence_scores = [1.0] * len(embeddings)
        if metadata is None:
            metadata = {}
        
        # Đảm bảo số lượng embeddings và confidence scores khớp nhau
        if len(embeddings) != len(confidence_scores):
            confidence_scores = [1.0] * len(embeddings)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Kiểm tra tên đã tồn tại chưa
            cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            existing = cursor.fetchone()
            
            if existing:
                # Cập nhật người đã có
                person_id = existing[0]
                self._update_person_embeddings(cursor, person_id, embeddings, 
                                             image_paths, confidence_scores, metadata)
            else:
                # Thêm người mới
                person_id = self._insert_new_person(cursor, name, embeddings,
                                                  image_paths, confidence_scores, metadata)
            
            conn.commit()
            return person_id
    
    def _insert_new_person(self, cursor, name: str, embeddings: List[np.ndarray],
                          image_paths: List[str], confidence_scores: List[float],
                          metadata: Dict[str, Any]) -> int:
        """Thêm người mới"""
        # Serialize embeddings
        embeddings_blob = pickle.dumps(embeddings)
        
        # Insert vào persons table
        cursor.execute('''
            INSERT INTO persons (name, embeddings, metadata, image_paths, confidence_scores)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, embeddings_blob, json.dumps(metadata), 
              json.dumps(image_paths), json.dumps(confidence_scores)))
        
        person_id = cursor.lastrowid
        
        # Insert từng embedding riêng biệt
        for i, (embedding, confidence, image_path) in enumerate(
            zip(embeddings, confidence_scores, image_paths + [""] * len(embeddings))
        ):
            embedding_blob = pickle.dumps(embedding)
            cursor.execute('''
                INSERT INTO embeddings (person_id, embedding, confidence, image_path)
                VALUES (?, ?, ?, ?)
            ''', (person_id, embedding_blob, confidence, image_path))
        
        return person_id
    
    def _update_person_embeddings(self, cursor, person_id: int, 
                                 new_embeddings: List[np.ndarray],
                                 new_image_paths: List[str],
                                 new_confidence_scores: List[float],
                                 new_metadata: Dict[str, Any]):
        """Cập nhật embeddings cho người đã có"""
        # Lấy dữ liệu hiện tại
        cursor.execute('SELECT embeddings, metadata, image_paths, confidence_scores FROM persons WHERE id = ?', (person_id,))
        result = cursor.fetchone()
        
        if result:
            old_embeddings = pickle.loads(result[0])
            old_metadata = json.loads(result[1]) if result[1] else {}
            old_image_paths = json.loads(result[2]) if result[2] else []
            old_confidence_scores = json.loads(result[3]) if result[3] else []
            
            # Merge embeddings
            all_embeddings = old_embeddings + new_embeddings
            all_image_paths = old_image_paths + new_image_paths
            all_confidence_scores = old_confidence_scores + new_confidence_scores
            
            # Merge metadata
            merged_metadata = {**old_metadata, **new_metadata}
            
            # Cập nhật persons table
            embeddings_blob = pickle.dumps(all_embeddings)
            cursor.execute('''
                UPDATE persons 
                SET embeddings = ?, metadata = ?, image_paths = ?, confidence_scores = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (embeddings_blob, json.dumps(merged_metadata), 
                  json.dumps(all_image_paths), json.dumps(all_confidence_scores), person_id))
            
            # Thêm embeddings mới vào embeddings table
            for embedding, confidence, image_path in zip(new_embeddings, new_confidence_scores, new_image_paths):
                embedding_blob = pickle.dumps(embedding)
                cursor.execute('''
                    INSERT INTO embeddings (person_id, embedding, confidence, image_path)
                    VALUES (?, ?, ?, ?)
                ''', (person_id, embedding_blob, confidence, image_path))
    
    def get_person(self, name: str) -> Optional[PersonRecord]:
        """Lấy thông tin người theo tên"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, embeddings, metadata, created_at, updated_at, image_paths, confidence_scores
                FROM persons WHERE name = ?
            ''', (name,))
            
            result = cursor.fetchone()
            if result:
                return PersonRecord(
                    id=result[0],
                    name=result[1],
                    embeddings=pickle.loads(result[2]),
                    metadata=json.loads(result[3]) if result[3] else {},
                    created_at=datetime.fromisoformat(result[4]),
                    updated_at=datetime.fromisoformat(result[5]),
                    image_paths=json.loads(result[6]) if result[6] else [],
                    confidence_scores=json.loads(result[7]) if result[7] else []
                )
            return None
    
    def get_person_by_id(self, person_id: int) -> Optional[PersonRecord]:
        """Lấy thông tin người theo ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, embeddings, metadata, created_at, updated_at, image_paths, confidence_scores
                FROM persons WHERE id = ?
            ''', (person_id,))
            
            result = cursor.fetchone()
            if result:
                return PersonRecord(
                    id=result[0],
                    name=result[1],
                    embeddings=pickle.loads(result[2]),
                    metadata=json.loads(result[3]) if result[3] else {},
                    created_at=datetime.fromisoformat(result[4]),
                    updated_at=datetime.fromisoformat(result[5]),
                    image_paths=json.loads(result[6]) if result[6] else [],
                    confidence_scores=json.loads(result[7]) if result[7] else []
                )
            return None
    
    def get_all_persons(self) -> List[PersonRecord]:
        """Lấy tất cả người trong database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, embeddings, metadata, created_at, updated_at, image_paths, confidence_scores
                FROM persons ORDER BY name
            ''')
            
            results = []
            for row in cursor.fetchall():
                results.append(PersonRecord(
                    id=row[0],
                    name=row[1],
                    embeddings=pickle.loads(row[2]),
                    metadata=json.loads(row[3]) if row[3] else {},
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    image_paths=json.loads(row[6]) if row[6] else [],
                    confidence_scores=json.loads(row[7]) if row[7] else []
                ))
            
            return results
    
    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Lấy tất cả embeddings dưới dạng dictionary"""
        persons = self.get_all_persons()
        embeddings_dict = {}
        
        for person in persons:
            embeddings_dict[person.name] = person.embeddings
        
        return embeddings_dict
    
    def delete_person(self, name: str) -> bool:
        """Xóa người khỏi database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Xóa embeddings trước
            cursor.execute('DELETE FROM embeddings WHERE person_id = (SELECT id FROM persons WHERE name = ?)', (name,))
            
            # Xóa person
            cursor.execute('DELETE FROM persons WHERE name = ?', (name,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def update_person_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Cập nhật metadata của người"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Lấy metadata hiện tại
            cursor.execute('SELECT metadata FROM persons WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result:
                old_metadata = json.loads(result[0]) if result[0] else {}
                merged_metadata = {**old_metadata, **metadata}
                
                cursor.execute('''
                    UPDATE persons 
                    SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                ''', (json.dumps(merged_metadata), name))
                
                conn.commit()
                return cursor.rowcount > 0
            
            return False
    
    def search_persons(self, query: str) -> List[PersonRecord]:
        """Tìm kiếm người theo tên"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, embeddings, metadata, created_at, updated_at, image_paths, confidence_scores
                FROM persons WHERE name LIKE ? ORDER BY name
            ''', (f'%{query}%',))
            
            results = []
            for row in cursor.fetchall():
                results.append(PersonRecord(
                    id=row[0],
                    name=row[1],
                    embeddings=pickle.loads(row[2]),
                    metadata=json.loads(row[3]) if row[3] else {},
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    image_paths=json.loads(row[6]) if row[6] else [],
                    confidence_scores=json.loads(row[7]) if row[7] else []
                ))
            
            return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Lấy thống kê database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Số lượng người
            cursor.execute('SELECT COUNT(*) FROM persons')
            person_count = cursor.fetchone()[0]
            
            # Số lượng embeddings
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            embedding_count = cursor.fetchone()[0]
            
            # Người mới nhất
            cursor.execute('SELECT name, created_at FROM persons ORDER BY created_at DESC LIMIT 1')
            latest_person = cursor.fetchone()
            
            # Kích thước database
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'person_count': person_count,
                'embedding_count': embedding_count,
                'latest_person': latest_person[0] if latest_person else None,
                'latest_created_at': latest_person[1] if latest_person else None,
                'database_size_mb': db_size / (1024 * 1024)
            }
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Backup database"""
        if backup_path is None:
            backup_path = self.backup_path
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        self.logger.info(f"Database đã được backup đến {backup_path}")
        return backup_path
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database từ backup"""
        try:
            import shutil
            shutil.copy2(backup_path, self.db_path)
            self.logger.info(f"Database đã được restore từ {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi restore database: {e}")
            return False
    
    def export_to_json(self, export_path: str) -> bool:
        """Export database ra JSON"""
        try:
            persons = self.get_all_persons()
            data = {
                'exported_at': datetime.now().isoformat(),
                'person_count': len(persons),
                'persons': [person.to_dict() for person in persons]
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Database đã được export đến {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi export database: {e}")
            return False
    
    def import_from_json(self, import_path: str) -> bool:
        """Import database từ JSON"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for person_data in data.get('persons', []):
                person = PersonRecord.from_dict(person_data)
                self.add_person(
                    name=person.name,
                    embeddings=person.embeddings,
                    image_paths=person.image_paths,
                    confidence_scores=person.confidence_scores,
                    metadata=person.metadata
                )
            
            self.logger.info(f"Database đã được import từ {import_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi import database: {e}")
            return False


# Utility functions
def create_database_hash(embeddings: Dict[str, List[np.ndarray]]) -> str:
    """Tạo hash cho database embeddings"""
    hash_string = ""
    for name in sorted(embeddings.keys()):
        for embedding in embeddings[name]:
            hash_string += f"{name}:{embedding.tobytes()}"
    
    return hashlib.md5(hash_string.encode()).hexdigest()


def compare_databases(db1: FaceDatabase, db2: FaceDatabase) -> Dict[str, Any]:
    """So sánh hai database"""
    stats1 = db1.get_database_stats()
    stats2 = db2.get_database_stats()
    
    persons1 = {p.name for p in db1.get_all_persons()}
    persons2 = {p.name for p in db2.get_all_persons()}
    
    return {
        'db1_stats': stats1,
        'db2_stats': stats2,
        'common_persons': persons1.intersection(persons2),
        'db1_only': persons1 - persons2,
        'db2_only': persons2 - persons1,
        'identical': persons1 == persons2
    }


# Example usage
if __name__ == "__main__":
    # Test database
    db = FaceDatabase("test_face_database.db")
    
    # Thêm người test
    test_embeddings = [np.random.randn(512) for _ in range(3)]
    person_id = db.add_person(
        name="Test Person",
        embeddings=test_embeddings,
        image_paths=["test1.jpg", "test2.jpg", "test3.jpg"],
        confidence_scores=[0.9, 0.8, 0.95],
        metadata={"age": 25, "gender": "male"}
    )
    
    print(f"Added person with ID: {person_id}")
    
    # Lấy thông tin
    person = db.get_person("Test Person")
    if person:
        print(f"Person: {person.name}")
        print(f"Embeddings count: {len(person.embeddings)}")
        print(f"Metadata: {person.metadata}")
    
    # Thống kê
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Export
    db.export_to_json("test_export.json")
    
    # Cleanup
    os.remove("test_face_database.db")
    os.remove("test_export.json")
