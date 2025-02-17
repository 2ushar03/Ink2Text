import React from 'react';
import { FaFileImage } from 'react-icons/fa';

const Navbar = () => {
  return (
    <nav className="bg-[#493D9E] rounded-b-xl p-4">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FaFileImage className="text-white text-2xl" />
          <span className="text-white text-xl font-bold">Image To Text</span>
        </div>
        <div className="flex space-x-4">
          
        </div>
      </div>
    </nav>
  );
};

export default Navbar;