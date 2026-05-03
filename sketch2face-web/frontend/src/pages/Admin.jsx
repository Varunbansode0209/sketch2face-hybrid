import { useState, useEffect } from 'react';
import { Users, Image, BarChart3, Upload, Trash2 } from 'lucide-react';
import { adminAPI } from '../api/admin.api';
import { API_BASE_URL } from '../api/axios';
import Loader from '../components/Loader';

const Admin = () => {
  const [activeTab, setActiveTab] = useState('stats');
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [faces, setFaces] = useState([]);
  const [galleryFilter, setGalleryFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [uploadType, setUploadType] = useState('photo'); // photo, sketch, both
  const [targetGallery, setTargetGallery] = useState('cufs'); // celeba or cufs
  const [photoFile, setPhotoFile] = useState(null);
  const [sketchFile, setSketchFile] = useState(null);
  const [personName, setPersonName] = useState('');
  const [description, setDescription] = useState('');
  const [gender, setGender] = useState('');
  const [age, setAge] = useState('');

  useEffect(() => {
    fetchStats();
    fetchUsers();
    fetchFaces();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await adminAPI.getStatistics();
      setStats(response);
    } catch (err) {
      console.error('Failed to fetch stats');
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await adminAPI.getAllUsers();
      setUsers(response.users || []);
    } catch (err) {
      console.error('Failed to fetch users');
    } finally {
      setLoading(false);
    }
  };

  const fetchFaces = async (filter = galleryFilter) => {
    try {
      const response = await adminAPI.getAllFaces(filter);
      setFaces(response.faces || []);
    } catch (err) {
      console.error('Failed to fetch faces');
    }
  };

  const handleUploadFace = async () => {
    if (uploadType === 'photo' && !photoFile) return alert('Please select a photo file');
    if (uploadType === 'sketch' && !sketchFile) return alert('Please select a sketch file');
    if (uploadType === 'both' && (!photoFile || !sketchFile)) return alert('Please select both a photo and a sketch');

    try {
      const uploadPromises = [];

      // Create base form data with text fields
      const createFormData = (file, gallery) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('gallery', gallery);
        if (personName) formData.append('name', personName);
        if (description) formData.append('description', description);
        if (gender) formData.append('gender', gender);
        if (age) formData.append('age', age);
        return formData;
      };

      if (uploadType === 'photo' || uploadType === 'both') {
        uploadPromises.push(adminAPI.uploadFace(createFormData(photoFile, targetGallery)));
      }

      if (uploadType === 'sketch' || uploadType === 'both') {
        uploadPromises.push(adminAPI.uploadFace(createFormData(sketchFile, targetGallery)));
      }

      await Promise.all(uploadPromises);
      
      alert('Record uploaded successfully!');
      
      // Reset form
      setPhotoFile(null);
      setSketchFile(null);
      setPersonName('');
      setDescription('');
      setGender('');
      setAge('');
      fetchFaces();
    } catch (err) {
      console.error(err);
      alert('Failed to upload record. Please try again.');
    }
  };

  const handleDeleteFace = async (faceId) => {
    if (!confirm('Are you sure you want to delete this face?')) return;
    
    try {
      await adminAPI.deleteFace(faceId);
      alert('Face deleted successfully');
      fetchFaces();
    } catch (err) {
      alert('Failed to delete face');
    }
  };

  const handleDeleteUser = async (userId) => {
    if (!confirm('Are you sure you want to delete this user?')) return;

    try {
      await adminAPI.deleteUser(userId);
      alert('User deleted successfully');
      fetchUsers();
    } catch (err) {
      alert('Failed to delete user');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader message="Loading admin panel..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-7xl">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">Admin Dashboard</h1>

        <div className="flex gap-4 mb-8 overflow-x-auto">
          <button
            onClick={() => setActiveTab('stats')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'stats'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <BarChart3 className="w-5 h-5" />
            Statistics
          </button>
          <button
            onClick={() => setActiveTab('users')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'users'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Users className="w-5 h-5" />
            Users
          </button>
          <button
            onClick={() => setActiveTab('faces')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'faces'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Upload className="w-5 h-5" />
            Upload Records
          </button>
          <button
            onClick={() => setActiveTab('manage_galleries')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition ${
              activeTab === 'manage_galleries'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Image className="w-5 h-5" />
            Manage Galleries
          </button>
        </div>

        {activeTab === 'stats' && stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <Users className="w-12 h-12 text-indigo-600 mb-4" />
              <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Users</h3>
              <p className="text-4xl font-bold text-gray-800">{stats.total_users || 0}</p>
            </div>
            <div className="bg-white rounded-lg shadow-lg p-6">
              <BarChart3 className="w-12 h-12 text-green-600 mb-4" />
              <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Matches</h3>
              <p className="text-4xl font-bold text-gray-800">{stats.total_queries || 0}</p>
            </div>
            <div className="bg-white rounded-lg shadow-lg p-6">
              <Image className="w-12 h-12 text-purple-600 mb-4" />
              <h3 className="text-gray-600 text-sm font-semibold mb-2">Faces in DB</h3>
              <p className="text-4xl font-bold text-gray-800">{stats.total_suspects || 0}</p>
            </div>
          </div>
        )}

        {activeTab === 'users' && (
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Joined</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {users.map((user) => (
                  <tr key={user.id}>
                    <td className="px-6 py-4 text-sm text-black font-semibold">{user.name}</td>
                    <td className="px-6 py-4 text-sm text-black">{user.email}</td>
                    <td className="px-6 py-4 text-sm">
                      <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        user.role === 'admin' 
                          ? 'bg-purple-100 text-purple-800' 
                          : user.role === 'investigator' 
                            ? 'bg-blue-100 text-blue-800' 
                            : 'bg-gray-100 text-gray-800'
                      }`}>
                        {user.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : 'Public'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-black font-medium">
                      {new Date(user.createdAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => handleDeleteUser(user.id)}
                        className="text-red-600 hover:text-red-800"
                        title="Delete User"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'faces' && (
          <div>
            <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden mb-8">
              <div className="bg-indigo-600 px-6 py-4">
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload New Record
                </h3>
                <p className="text-indigo-100 text-sm mt-1">Add suspects into the global database. Changes reflect instantly.</p>
              </div>
              
              <div className="p-6">
                {/* Upload Settings Matrix */}
                <div className="mb-6 pb-6 border-b border-gray-100 flex flex-col md:flex-row gap-8">
                  {/* Upload Type Selector */}
                  <div className="flex-1">
                    <label className="block text-sm font-semibold text-gray-700 mb-3">What are you uploading?</label>
                    <div className="flex flex-wrap gap-4">
                      <button
                        onClick={() => setUploadType('photo')}
                        className={`flex-1 min-w-[140px] py-3 px-4 rounded-lg border-2 transition ${
                          uploadType === 'photo' ? 'border-indigo-600 bg-indigo-50 text-indigo-700' : 'border-gray-200 text-gray-600 hover:border-gray-300'
                        } font-medium`}
                      >
                        Photo Only
                      </button>
                      <button
                        onClick={() => setUploadType('sketch')}
                        className={`flex-1 min-w-[140px] py-3 px-4 rounded-lg border-2 transition ${
                          uploadType === 'sketch' ? 'border-indigo-600 bg-indigo-50 text-indigo-700' : 'border-gray-200 text-gray-600 hover:border-gray-300'
                        } font-medium`}
                      >
                        Sketch Only
                      </button>
                      <button
                        onClick={() => setUploadType('both')}
                        className={`flex-1 min-w-[140px] py-3 px-4 rounded-lg border-2 transition ${
                          uploadType === 'both' ? 'border-indigo-600 bg-indigo-50 text-indigo-700' : 'border-gray-200 text-gray-600 hover:border-gray-300'
                        } font-medium`}
                      >
                        Both (Photo & Sketch)
                      </button>
                    </div>
                  </div>

                  {/* Target Gallery Selector */}
                  <div className="md:w-64">
                    <label className="block text-sm font-semibold text-gray-700 mb-3">Target Gallery Database</label>
                    <div className="flex flex-col gap-3">
                      <select
                        value={targetGallery}
                        onChange={(e) => setTargetGallery(e.target.value)}
                        className="w-full border-2 border-indigo-200 rounded-lg px-4 py-3 bg-indigo-50/50 text-black font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-600 transition"
                      >
                        <option value="cufs" className="font-semibold text-black">CUFS (Forensic Data)</option>
                        <option value="celeba" className="font-semibold text-black">CelebA (Public Faces)</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
                  {/* File Upload Regions */}
                  <div className="space-y-4">
                    <label className="block text-sm font-semibold text-gray-700">Image Files</label>
                    
                    {(uploadType === 'photo' || uploadType === 'both') && (
                      <div className="relative border-2 border-dashed border-gray-300 rounded-lg p-4 hover:bg-gray-50 transition cursor-pointer flex flex-col items-center justify-center min-h-[120px]">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => setPhotoFile(e.target.files[0])}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        <Image className="w-8 h-8 text-gray-400 mb-2" />
                        <span className="text-sm font-medium text-gray-600">
                          {photoFile ? photoFile.name : 'Select Photo File'}
                        </span>
                        {!photoFile && <span className="text-xs text-gray-400 mt-1">JPEG, PNG, JPG</span>}
                      </div>
                    )}
                    
                    {(uploadType === 'sketch' || uploadType === 'both') && (
                      <div className="relative border-2 border-dashed border-gray-300 rounded-lg p-4 hover:bg-gray-50 transition cursor-pointer flex flex-col items-center justify-center min-h-[120px]">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => setSketchFile(e.target.files[0])}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        <Image className="w-8 h-8 text-indigo-400 mb-2" />
                        <span className="text-sm font-medium text-indigo-600">
                          {sketchFile ? sketchFile.name : 'Select Sketch File'}
                        </span>
                        {!sketchFile && <span className="text-xs text-gray-400 mt-1">JPEG, PNG, JPG</span>}
                      </div>
                    )}
                  </div>

                  {/* Metadata Fields */}
                  <div className="space-y-4">
                    <label className="block text-sm font-semibold text-gray-700">Suspect Details</label>
                    
                    <input
                      type="text"
                      placeholder="Person Name (Optional)"
                      value={personName}
                      onChange={(e) => setPersonName(e.target.value)}
                      className="w-full border border-gray-300 rounded-lg px-4 py-3 text-black font-semibold placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                    
                    <textarea
                      placeholder="Description (e.g. contextual details, appearance)"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      className="w-full border border-gray-300 rounded-lg px-4 py-3 text-black font-semibold placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none h-24"
                    />
                    
                    <div className="grid grid-cols-2 gap-4">
                      <select
                        value={gender}
                        onChange={(e) => setGender(e.target.value)}
                        className={`w-full border border-gray-300 rounded-lg px-4 py-3 font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500 ${gender === '' ? 'text-gray-500' : 'text-black'}`}
                      >
                        <option value="">Select Gender</option>
                        <option value="Male" className="text-black">Male</option>
                        <option value="Female" className="text-black">Female</option>
                        <option value="Other" className="text-black">Other</option>
                      </select>
                      
                      <input
                        type="number"
                        placeholder="Age"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        className="w-full border border-gray-300 rounded-lg px-4 py-3 text-black font-semibold placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-100 flex justify-end">
                  <button
                    onClick={handleUploadFace}
                    className="bg-indigo-600 shadow-md text-white px-8 py-3 rounded-lg hover:bg-indigo-700 hover:shadow-lg transition flex items-center gap-2 font-semibold"
                  >
                    <Upload className="w-5 h-5" />
                    Submit Record To Database
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'manage_galleries' && (
          <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 overflow-hidden">
            <div className="flex items-center justify-between mb-6 pb-6 border-b border-gray-100">
              <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
                <Image className="w-6 h-6 text-indigo-600" />
                Live Database Directory
              </h2>
            </div>

            {/* Gallery Filter Tracker */}
            <div className="flex items-center justify-end mb-4 px-2">
              <div className="flex bg-gray-200 p-1 rounded-lg">
                <button
                  onClick={() => { setGalleryFilter('all'); fetchFaces('all'); }}
                  className={`px-4 py-2 rounded-md font-bold text-sm transition-all ${galleryFilter === 'all' ? 'bg-white text-black shadow-sm' : 'text-gray-500 hover:text-black'}`}
                >
                  All Galleries
                </button>
                <button
                  onClick={() => { setGalleryFilter('cufs'); fetchFaces('cufs'); }}
                  className={`px-4 py-2 rounded-md font-bold text-sm transition-all ${galleryFilter === 'cufs' ? 'bg-white text-black shadow-sm' : 'text-gray-500 hover:text-black'}`}
                >
                  CUFS (Sketches)
                </button>
                <button
                  onClick={() => { setGalleryFilter('celeba'); fetchFaces('celeba'); }}
                  className={`px-4 py-2 rounded-md font-bold text-sm transition-all ${galleryFilter === 'celeba' ? 'bg-white text-black shadow-sm' : 'text-gray-500 hover:text-black'}`}
                >
                  CelebA (Photos)
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Image</th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Name / Identity</th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Demographics</th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Source Gallery</th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Date Added</th>
                    <th className="px-6 py-4 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 border-t border-gray-200">
                  {faces.map((face) => {
                    let imageUrl = face.image_path;
                    if (imageUrl && !imageUrl.startsWith('http')) {
                      const filename = imageUrl.split(/[\\/]/).pop();
                      imageUrl = `${API_BASE_URL}/gallery/${face.gallery}/${filename}`;
                    }
                    
                    return (
                      <tr key={face.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <img
                            src={imageUrl || '/placeholder.jpg'}
                            alt="Record"
                            className="w-14 h-14 object-cover rounded-md shadow-sm border border-gray-200"
                          />
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <p className="text-sm font-bold text-black">{face.name || 'Unnamed Suspect Record'}</p>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-black">
                            {face.gender || 'Unknown'} <span className="text-gray-400 mx-1">|</span> {face.age ? `Age: ${face.age}` : 'Age: N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {face.gallery === 'celeba' ? (
                            <span className="px-3 py-1 inline-flex text-xs leading-5 font-bold rounded-full bg-blue-100 text-blue-800">
                              CelebA (Photos)
                            </span>
                          ) : (
                            <span className="px-3 py-1 inline-flex text-xs leading-5 font-bold rounded-full bg-purple-100 text-purple-800">
                              CUFS (Sketches)
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-600">
                          {face.created_at ? new Date(face.created_at).toLocaleDateString() : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <button
                            onClick={() => handleDeleteFace(face.id)}
                            className="inline-flex items-center gap-2 bg-red-50 text-red-600 hover:bg-red-100 hover:text-red-700 px-3 py-2 rounded-lg transition-colors font-bold"
                            title="Delete Record"
                          >
                            <Trash2 className="w-4 h-4" />
                            Delete
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {faces.length === 0 && (
                <div className="py-16 text-center">
                  <Image className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500 font-bold text-lg">No records found mapped to this gallery view.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Admin;