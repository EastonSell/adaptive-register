import hashlib,io,tempfile,unittest
from pathlib import Path
try: import fetch_model
except ImportError: fetch_model=None

class FetchTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(fetch_model,'download helper is not implemented')
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)/'model'
  self.blob=b'fixture data';self.manifest={'model_id':'owner/model','revision':'a'*40,'files':{'config.json':{'bytes':len(self.blob),'sha256':hashlib.sha256(self.blob).hexdigest()}}}
 def test_downloads_verified_bytes(self):
  urls=[]
  def open_fixture(url):urls.append(url);return io.BytesIO(self.blob)
  fetch_model.fetch(self.manifest,self.out,opener=open_fixture)
  self.assertEqual((self.out/'config.json').read_bytes(),self.blob)
  self.assertEqual(urls,['https://huggingface.co/owner/model/resolve/'+'a'*40+'/config.json'])
 def test_existing_valid_file_needs_no_network(self):
  self.out.mkdir();(self.out/'config.json').write_bytes(self.blob)
  fetch_model.fetch(self.manifest,self.out,opener=lambda _:self.fail('network called'))
 def test_existing_wrong_file_not_overwritten(self):
  self.out.mkdir();f=self.out/'config.json';f.write_bytes(b'old')
  with self.assertRaises(ValueError):fetch_model.fetch(self.manifest,self.out)
  self.assertEqual(f.read_bytes(),b'old')
 def test_bad_download_not_published(self):
  with self.assertRaises(ValueError):fetch_model.fetch(self.manifest,self.out,opener=lambda _:io.BytesIO(b'x'*len(self.blob)))
  self.assertEqual(list(self.out.iterdir()),[])
 def test_rejects_unsafe_paths_and_unpinned_revision(self):
  for name,revision in [('../outside','a'*40),('code.py','a'*40),('config.json','main')]:
   m=dict(self.manifest,revision=revision,files={name:next(iter(self.manifest['files'].values()))})
   with self.assertRaises(ValueError):fetch_model.fetch(m,self.out,opener=lambda _:self.fail('network called'))
 def test_verify_only_missing_does_not_download(self):
  with self.assertRaises(FileNotFoundError):fetch_model.fetch(self.manifest,self.out,verify_only=True,opener=lambda _:self.fail('network called'))
  self.assertFalse(self.out.exists())
